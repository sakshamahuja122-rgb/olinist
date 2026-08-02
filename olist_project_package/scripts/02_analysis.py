"""
02_analysis.py
---------------
End-to-end churn diagnosis for the Olist retention case.

Phase 1 (Descriptive):  cohort retention curves, RFM segmentation
Phase 2 (Diagnostic):   logistic regression + random forest churn drivers
Phase 3 (Prescriptive): scenario modeling -> revenue impact of fixing delivery

Outputs:
  outputs/results.json         -> every number used in the exec summary/dashboard
  outputs/*.png                -> charts for the Word doc
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DATA = "/home/claude/olist_project/data"
OUT = "/home/claude/olist_project/outputs"
plt.rcParams["figure.dpi"] = 150
COLOR = "#2b4570"
ACCENT = "#e07a5f"

# ---------------------------------------------------------------------------
# 0. LOAD + JOIN  (this is the "SQL/data engineering" step from raw tables)
# ---------------------------------------------------------------------------
orders = pd.read_csv(f"{DATA}/orders.csv", parse_dates=[
    "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
orders = orders.drop(columns=["customer_state"])  # authoritative version lives on customers table
items = pd.read_csv(f"{DATA}/order_items.csv")
reviews = pd.read_csv(f"{DATA}/order_reviews.csv")
customers = pd.read_csv(f"{DATA}/customers.csv")

df = (orders
      .merge(items, on="order_id", how="left")
      .merge(reviews, on="order_id", how="left")
      .merge(customers, on="customer_unique_id", how="left"))

df["delay_vs_estimate_days"] = (
    (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400
)
df["freight_ratio"] = df["freight_value"] / df["price"]
df = df.sort_values(["customer_unique_id", "order_purchase_timestamp"])
df["order_seq"] = df.groupby("customer_unique_id").cumcount() + 1
df["cohort_month"] = df.groupby("customer_unique_id")["order_purchase_timestamp"].transform("min").dt.to_period("M")

first_orders = df[df["order_seq"] == 1].copy()
repeat_customers = set(df.loc[df["order_seq"] == 2, "customer_unique_id"])
first_orders["churned"] = (~first_orders["customer_unique_id"].isin(repeat_customers)).astype(int)

results = {}
results["n_customers"] = int(df["customer_unique_id"].nunique())
results["n_orders"] = int(len(df))
results["repeat_purchase_rate"] = float(1 - first_orders["churned"].mean())
results["churn_rate"] = float(first_orders["churned"].mean())

# ---------------------------------------------------------------------------
# PHASE 1a: COHORT RETENTION
# ---------------------------------------------------------------------------
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")
df["month_index"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)
cohort_counts = df.groupby(["cohort_month", "month_index"])['customer_unique_id'].nunique().reset_index()
cohort_sizes = first_orders.groupby("cohort_month")["customer_unique_id"].nunique()
cohort_pivot = cohort_counts.pivot(index="cohort_month", columns="month_index", values="customer_unique_id")
retention_pct = cohort_pivot.divide(cohort_sizes, axis=0)

overall_retention_curve = retention_pct.mean(axis=0).dropna()
overall_retention_curve = overall_retention_curve[overall_retention_curve.index >= 0]
results["retention_curve"] = {int(k): round(float(v) * 100, 2) for k, v in overall_retention_curve.items() if k <= 6}

plt.figure(figsize=(7, 4.2))
months = list(results["retention_curve"].keys())
vals = list(results["retention_curve"].values())
plt.plot(months, vals, marker="o", color=COLOR, linewidth=2.5)
plt.fill_between(months, vals, color=COLOR, alpha=0.08)
plt.title("Customer Retention Curve (avg. across acquisition cohorts)", fontsize=11, fontweight="bold")
plt.xlabel("Months since first purchase")
plt.ylabel("% of cohort still active")
plt.ylim(0, max(vals) * 1.3 if vals else 10)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/cohort_retention.png")
plt.close()

# ---------------------------------------------------------------------------
# PHASE 1b: RFM SEGMENTATION
# ---------------------------------------------------------------------------
snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
rfm = df.groupby("customer_unique_id").agg(
    recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("price", "sum"),
).reset_index()

rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
rfm["f_score"] = pd.cut(rfm["frequency"], bins=[0, 1, 2, 3, 100], labels=[1, 2, 3, 4]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["rfm_sum"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]


def segment(row):
    if row["frequency"] == 1:
        return "One-Time Buyer (At Risk of Never Returning)"
    if row["rfm_sum"] >= 10:
        return "Champions"
    if row["rfm_sum"] >= 7:
        return "Loyal / Steady"
    return "Lost / Lapsed Repeat Buyer"


rfm["segment"] = rfm.apply(segment, axis=1)
seg_counts = rfm["segment"].value_counts()
seg_revenue = rfm.groupby("segment")["monetary"].sum().sort_values(ascending=False)
results["rfm_segments"] = {
    seg: {"customers": int(seg_counts[seg]), "revenue": round(float(seg_revenue[seg]), 2)}
    for seg in seg_counts.index
}

plt.figure(figsize=(7, 4.2))
seg_counts_sorted = seg_counts.sort_values()
plt.barh(seg_counts_sorted.index, seg_counts_sorted.values, color=COLOR)
plt.title("Customer Base by RFM Segment", fontsize=11, fontweight="bold")
plt.xlabel("# Customers")
plt.tight_layout()
plt.savefig(f"{OUT}/rfm_segments.png")
plt.close()

# ---------------------------------------------------------------------------
# PHASE 2: DIAGNOSTIC — what DRIVES churn?
# ---------------------------------------------------------------------------
cat_defect = df.groupby("product_category_name")["review_score"].apply(lambda s: (s <= 2).mean())
first_orders["category_defect_rate"] = first_orders["product_category_name"].map(cat_defect)

feat_cols = ["delay_vs_estimate_days", "freight_ratio", "review_score", "category_defect_rate", "price"]
model_df = first_orders.dropna(subset=feat_cols + ["churned"]).copy()

X = model_df[feat_cols]
y = model_df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

logit = LogisticRegression(max_iter=1000)
logit.fit(X_train_s, y_train)
logit_auc = roc_auc_score(y_test, logit.predict_proba(X_test_s)[:, 1])

rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=30, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

# Permutation importance on the RF model (SHAP unavailable offline; permutation
# importance gives an equivalent "which feature moves the prediction" ranking)
perm = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=42, scoring="roc_auc")
importance = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)

results["model"] = {
    "logistic_auc": round(float(logit_auc), 3),
    "random_forest_auc": round(float(rf_auc), 3),
    "feature_importance": {k: round(float(v), 4) for k, v in importance.items()},
    "logit_coefficients": {feat_cols[i]: round(float(c), 3) for i, c in enumerate(logit.coef_[0])},
}

plt.figure(figsize=(7, 4.2))
imp_sorted = importance.sort_values()
labels_map = {
    "delay_vs_estimate_days": "Delivery delay vs. promise",
    "freight_ratio": "Freight cost ratio",
    "review_score": "Review score",
    "category_defect_rate": "Product category defect rate",
    "price": "Order price",
}
plt.barh([labels_map.get(i, i) for i in imp_sorted.index], imp_sorted.values, color=ACCENT)
plt.title("Churn Driver Importance (permutation importance, RF model)", fontsize=11, fontweight="bold")
plt.xlabel("Importance (drop in AUC when shuffled)")
plt.tight_layout()
plt.savefig(f"{OUT}/driver_importance.png")
plt.close()

# missed-expectation cut effect (for the headline stat)
missed = model_df["delay_vs_estimate_days"] > 3
churn_missed = model_df.loc[missed, "churned"].mean()
churn_ontime = model_df.loc[~missed, "churned"].mean()
results["churn_rate_missed_delivery"] = round(float(churn_missed) * 100, 1)
results["churn_rate_ontime_delivery"] = round(float(churn_ontime) * 100, 1)
results["pct_churn_uplift_from_missed_delivery"] = round(
    float((churn_missed - churn_ontime) / churn_ontime * 100), 1)

# category churn rates
cat_churn = model_df.groupby("product_category_name")["churned"].mean().sort_values(ascending=False)
results["top_churn_categories"] = {k: round(float(v) * 100, 1) for k, v in cat_churn.head(10).items()}

# state delay
state_delay = df.groupby("customer_state")["delay_vs_estimate_days"].mean().sort_values(ascending=False)
results["state_avg_delay_days"] = {k: round(float(v), 2) for k, v in state_delay.items()}

plt.figure(figsize=(7, 4.2))
top_cat = cat_churn.head(10).sort_values()
plt.barh(top_cat.index, top_cat.values * 100, color=COLOR)
plt.title("Churn Rate by Product Category (Top 10)", fontsize=11, fontweight="bold")
plt.xlabel("% of first-time buyers who churned")
plt.tight_layout()
plt.savefig(f"{OUT}/category_churn.png")
plt.close()

# ---------------------------------------------------------------------------
# PHASE 3: PRESCRIPTIVE — scenario modeling / $ impact
# ---------------------------------------------------------------------------
avg_order_value = float(model_df["price"].mean())
avg_orders_per_repeat_customer = float(df[df["customer_unique_id"].isin(repeat_customers)]
                                        .groupby("customer_unique_id").size().mean())
ltv_repeat = avg_order_value * avg_orders_per_repeat_customer

n_missed = int(missed.sum())
# Scenario: cut "missed by >3 days" incidents by 30% via tiered shipping buffer
reduction_pct = 0.30
orders_saved_from_missing = n_missed * reduction_pct
uplift_in_return_prob = churn_ontime - churn_missed  # negative of churn diff = return-prob gain
# some of those recovered orders convert to repeat buyers at the "on time" rate
incremental_repeat_customers = orders_saved_from_missing * abs(uplift_in_return_prob)
annual_revenue_lift = incremental_repeat_customers * ltv_repeat

results["scenario"] = {
    "avg_order_value": round(avg_order_value, 2),
    "avg_orders_per_repeat_customer": round(avg_orders_per_repeat_customer, 2),
    "implied_ltv_repeat_customer": round(ltv_repeat, 2),
    "orders_missed_gt_3days": n_missed,
    "assumed_reduction_in_missed_pct": reduction_pct * 100,
    "incremental_repeat_customers": round(incremental_repeat_customers, 0),
    "estimated_annual_revenue_lift": round(annual_revenue_lift, 0),
}

# Revenue at risk = customers currently churned who missed delivery, valued at implied LTV
revenue_at_risk = int(missed[model_df["churned"] == 1].sum()) * ltv_repeat
results["revenue_at_risk"] = round(revenue_at_risk, 0)

# ---------------------------------------------------------------------------
# Extra client-facing KPIs
# ---------------------------------------------------------------------------
results["order_defect_rate_pct"] = round(float((df["review_score"] <= 2).mean()) * 100, 1)

# Repeat purchase rate within fixed windows (days), computed directly on dates
snap = df.merge(first_orders[["customer_unique_id", "order_purchase_timestamp"]]
                 .rename(columns={"order_purchase_timestamp": "first_purchase"}),
                 on="customer_unique_id", how="left")
snap["days_since_first"] = (snap["order_purchase_timestamp"] - snap["first_purchase"]).dt.days
rpr = {}
for window in [90, 180, 365]:
    returned = snap[(snap["order_seq"] == 2) & (snap["days_since_first"] <= window)]["customer_unique_id"].nunique()
    rpr[window] = round(returned / results["n_customers"] * 100, 2)
results["repeat_purchase_rate_by_window_pct"] = rpr

# Illustrative CAC assumption (not in the transactional data — a typical
# marketplace CAC benchmark) to express the LTV:CAC ratio the CFO cares about
ASSUMED_CAC = 45.0
results["assumed_cac"] = ASSUMED_CAC
results["ltv_cac_ratio_repeat_customer"] = round(ltv_repeat / ASSUMED_CAC, 2)
results["ltv_cac_ratio_onetime_customer"] = round(avg_order_value / ASSUMED_CAC, 2)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))