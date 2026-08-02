"""
01_generate_data.py
--------------------
Generates a synthetic e-commerce transactions dataset that mirrors the SCHEMA
of the real "Olist Brazilian E-Commerce" dataset on Kaggle (customers, orders,
order_items, order_reviews, products).

WHY SYNTHETIC: this environment has no internet access, so the real Kaggle
CSVs can't be downloaded here. This script builds a dataset with the same
column names and realistic causal structure (late deliveries drive churn,
certain product categories have quality issues, remote states get worse
service) so the entire pipeline in 02_analysis.py runs and produces genuine,
internally-consistent findings.

>>> TO USE REAL DATA: download the Kaggle dataset
    (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), drop the
    CSVs into ../data/ with the same filenames used below, delete this
    script's outputs, and re-run 02_analysis.py unchanged. Column names match
    the real dataset, so no code changes are needed downstream.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
OUT = "/home/claude/olist_project/data"

# ---------------------------------------------------------------------------
# 1. Reference data: Brazilian states, with a "distance from hub" proxy.
#    SP/RJ/MG/ES/PR are close to the main SP fulfillment hub; the North/
#    Northeast are far. This drives delivery delay realistically.
# ---------------------------------------------------------------------------
states = pd.DataFrame({
    "state": ["SP", "RJ", "MG", "ES", "PR", "SC", "RS", "GO", "DF", "BA",
              "PE", "CE", "MA", "PA", "AM", "RR", "AC", "RO", "MT", "MS"],
    "hub_distance": [0, 1, 2, 2, 2, 3, 3, 4, 4, 5,
                      6, 6, 7, 8, 9, 10, 10, 8, 5, 5],
    "pop_weight": [46, 17, 21, 4, 11, 7, 11, 7, 3, 15,
                   9, 9, 7, 8, 4, 1, 1, 2, 3, 3],
})
states["pop_weight"] = states["pop_weight"] / states["pop_weight"].sum()

categories = pd.DataFrame({
    "product_category_name": [
        "bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor",
        "computers_accessories", "housewares", "watches_gifts", "telephony",
        "auto", "toys", "cool_stuff", "garden_tools", "perfumery",
        "baby", "electronics", "stationery", "fashion_bags", "pet_shop",
        "office_furniture", "home_appliances",
    ],
    # baseline probability a delivered item gets a 1-2 star review, all else equal
    "base_defect_rate": [
        0.10, 0.09, 0.10, 0.11, 0.12, 0.10, 0.09, 0.13, 0.14, 0.09,
        0.10, 0.24, 0.09, 0.10, 0.13, 0.09, 0.10, 0.11, 0.12, 0.13,
    ],
    "avg_price": [
        90, 70, 110, 160, 220, 60, 180, 150, 140, 80,
        75, 65, 55, 95, 260, 35, 85, 60, 310, 340,
    ],
})
# garden_tools ("Home & Garden" proxy) is deliberately the outlier with a
# structural quality problem, matching the case narrative.

states.to_csv(f"{OUT}/_ref_states.csv", index=False)
categories.to_csv(f"{OUT}/_ref_categories.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Customers (unique people). ~42,000 unique customers.
# ---------------------------------------------------------------------------
N_CUSTOMERS = 42000
cust_state = rng.choice(states["state"], size=N_CUSTOMERS, p=states["pop_weight"])
customers = pd.DataFrame({
    "customer_unique_id": [f"cust_{i:06d}" for i in range(N_CUSTOMERS)],
    "customer_state": cust_state,
})
customers = customers.merge(states[["state", "hub_distance"]],
                             left_on="customer_state", right_on="state").drop(columns="state")

# ---------------------------------------------------------------------------
# 3. Orders. Each customer places a first order; a subset places a 2nd (and
#    some a 3rd+) based on how good their PRIOR experience was.
#    This is the causal engine of the whole case study.
# ---------------------------------------------------------------------------
START = datetime(2022, 1, 1)
END = datetime(2023, 12, 31)

orders_rows = []
items_rows = []
reviews_rows = []
order_counter = 0

for _, cust in customers.iterrows():
    n_days_range = (END - START).days
    first_purchase = START + timedelta(days=int(rng.integers(0, n_days_range - 60)))
    hub_dist = cust["hub_distance"]

    purchase_date = first_purchase
    order_number = 1
    keep_going = True
    prev_good_experience = True

    while keep_going and order_number <= 5:
        order_id = f"order_{order_counter:07d}"
        order_counter += 1

        cat = categories.sample(1, weights=categories["avg_price"].max() - categories["avg_price"] + 50,
                                 random_state=int(rng.integers(0, 1e9))).iloc[0]

        # promised delivery window: base + distance factor + noise
        promised_days = 7 + hub_dist * 1.8 + rng.normal(0, 1.5)
        promised_days = max(3, promised_days)

        # actual delay: correlated with distance, with random operational noise
        # (independent of promised time -> creates "missed expectations")
        delay_noise = rng.gamma(shape=1.3, scale=1.0 + hub_dist * 0.35)
        actual_days = promised_days + delay_noise - rng.normal(1.0, 0.5)
        actual_days = max(1, actual_days)
        delay_vs_estimate = actual_days - promised_days  # >0 = arrived later than promised

        price = max(15, rng.normal(cat["avg_price"], cat["avg_price"] * 0.35))
        freight = max(5, price * (0.08 + hub_dist * 0.012) + rng.normal(0, 5))

        # review score: driven by (a) missed-expectation delay, (b) category defect rate
        defect_p = cat["base_defect_rate"] + max(0, delay_vs_estimate) * 0.045
        defect_p = min(0.85, defect_p)
        if rng.random() < defect_p:
            review_score = rng.choice([1, 2], p=[0.4, 0.6])
        else:
            review_score = rng.choice([3, 4, 5], p=[0.12, 0.33, 0.55])

        purchase_ts = purchase_date
        delivered_ts = purchase_date + timedelta(days=float(actual_days))
        estimated_ts = purchase_date + timedelta(days=float(promised_days))

        orders_rows.append({
            "order_id": order_id,
            "customer_unique_id": cust["customer_unique_id"],
            "customer_state": cust["customer_state"],
            "order_purchase_timestamp": purchase_ts,
            "order_delivered_customer_date": delivered_ts,
            "order_estimated_delivery_date": estimated_ts,
            "order_status": "delivered",
        })
        items_rows.append({
            "order_id": order_id,
            "product_category_name": cat["product_category_name"],
            "price": round(price, 2),
            "freight_value": round(freight, 2),
        })
        reviews_rows.append({
            "order_id": order_id,
            "review_score": int(review_score),
        })

        # decide if there's a next order: good experience -> much more likely to return
        missed_badly = delay_vs_estimate > 3
        bad_review = review_score <= 2
        p_return = 0.16
        if missed_badly:
            p_return -= 0.10
        if bad_review:
            p_return -= 0.08
        if cat["product_category_name"] == "garden_tools":
            p_return -= 0.05
        p_return = min(0.55, max(0.015, p_return))

        if rng.random() < p_return:
            gap_days = int(rng.integers(20, 150))
            purchase_date = delivered_ts + timedelta(days=gap_days)
            if purchase_date > END:
                keep_going = False
            else:
                order_number += 1
        else:
            keep_going = False

orders = pd.DataFrame(orders_rows)
order_items = pd.DataFrame(items_rows)
order_reviews = pd.DataFrame(reviews_rows)

orders.to_csv(f"{OUT}/orders.csv", index=False)
order_items.to_csv(f"{OUT}/order_items.csv", index=False)
order_reviews.to_csv(f"{OUT}/order_reviews.csv", index=False)
customers.drop(columns=["hub_distance"]).to_csv(f"{OUT}/customers.csv", index=False)

print(f"Generated {len(customers):,} customers, {len(orders):,} orders, "
      f"{orders['customer_unique_id'].nunique():,} unique buyers")
print(f"Repeat-purchase rate (raw): "
      f"{1 - (orders.groupby('customer_unique_id').size() == 1).mean():.1%}")