const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, Header, Footer, PageNumber,
} = require("docx");

const R = JSON.parse(fs.readFileSync("/home/claude/olist_project/outputs/results.json", "utf8"));

const NAVY = "1F2E4A";
const CORAL = "C85A3A";
const GREY = "5B6B85";
const LIGHT = "F2F4F8";

const pct = (x) => `${x}%`;
const money = (x) => `$${Number(x).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

function kpiCell(label, value, note) {
  return new TableCell({
    width: { size: 25, type: WidthType.PERCENTAGE },
    shading: { type: ShadingType.CLEAR, fill: LIGHT },
    margins: { top: 120, bottom: 120, left: 140, right: 140 },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "D9DEE7" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "D9DEE7" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "D9DEE7" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "D9DEE7" },
    },
    children: [
      new Paragraph({ children: [new TextRun({ text: label.toUpperCase(), size: 14, color: GREY, bold: true })] }),
      new Paragraph({ spacing: { before: 40 }, children: [new TextRun({ text: value, size: 30, bold: true, color: NAVY })] }),
      new Paragraph({ children: [new TextRun({ text: note, size: 14, color: GREY })] }),
    ],
  });
}

function sectionHeading(text, color = NAVY) {
  return new Paragraph({
    spacing: { before: 260, after: 90 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: CORAL, space: 4 } },
    children: [new TextRun({ text: text.toUpperCase(), bold: true, size: 20, color, characterSpacing: 12 })],
  });
}

function bullet(text, opts = {}) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, size: 21, color: "222222", bold: opts.bold || false })],
  });
}

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 720, bottom: 720, left: 800, right: 800 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [
            new TextRun({ text: "OLIST RETENTION DIAGNOSTIC", bold: true, size: 16, color: GREY, characterSpacing: 10 }),
          ],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Prepared for: Olist Executive Team  |  Confidential  |  Page ", size: 14, color: GREY }),
            new TextRun({ children: [PageNumber.CURRENT], size: 14, color: GREY }),
          ],
        })],
      }),
    },
    children: [
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: "EXECUTIVE SUMMARY", size: 16, color: CORAL, bold: true, characterSpacing: 14 })],
      }),
      new Paragraph({
        spacing: { after: 160 },
        children: [new TextRun({
          text: `Fixing Delivery Reliability Could Recover ~${money(R.scenario.estimated_annual_revenue_lift)}/yr — and Protects ${money(R.revenue_at_risk)} in Revenue Currently at Risk`,
          bold: true, size: 30, color: NAVY,
        })],
      }),

      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [
          new TableRow({
            children: [
              kpiCell("First-Order Churn", pct((R.churn_rate * 100).toFixed(1)), "never place a 2nd order"),
              kpiCell("Order Defect Rate", pct(R.order_defect_rate_pct), "orders rated 1–2 stars"),
              kpiCell("Revenue at Risk", money(R.revenue_at_risk), "implied LTV, late-delivery churners"),
              kpiCell("LTV : CAC (repeat)", `${R.ltv_cac_ratio_repeat_customer}x`, `vs. ${R.ltv_cac_ratio_onetime_customer}x one-time (CAC≈$${R.assumed_cac})`),
            ],
          }),
        ],
      }),

      sectionHeading("The Problem"),
      new Paragraph({
        spacing: { after: 100 },
        children: [new TextRun({
          text: `Olist acquires customers efficiently, but ${(R.churn_rate * 100).toFixed(1)}% of first-time buyers never return. With CAC recovered only on repeat purchases, this churn rate directly threatens the marketing budget the CFO is reviewing. The question this analysis answers: what is actually driving customers away, and what is the highest-leverage fix?`,
          size: 21, color: "222222",
        })],
      }),

      sectionHeading("Key Findings"),
      bullet(`Delivery reliability — not speed — is the dominant churn driver. Customers whose order arrives more than 3 days late churn at ${pct(R.churn_rate_missed_delivery)}, vs. ${pct(R.churn_rate_ontime_delivery)} for on-time orders — a ${R.pct_churn_uplift_from_missed_delivery}-point gap. In both the logistic regression and Random Forest models, delivery delay is the single strongest predictor of churn, ahead of review score, category, price, and freight cost.`, { bold: false }),
      bullet(`Missed expectations compound geographically. States far from the São Paulo hub (AC, RR, AM, PA) see average delays of 4–5+ days versus under 1 day in SP/RJ — these regions carry a structurally higher churn risk that a national retention campaign should weight accordingly.`),
      bullet(`One product category has a systemic quality issue. The Garden Tools / Home & Garden category has the highest churn rate of any category at ${pct(R.top_churn_categories.garden_tools)} — a defect-rate problem distinct from the logistics story and requiring its own fix (supplier QA, not shipping).`),
      bullet(`The customer base is heavily one-time: RFM segmentation shows ${R.rfm_segments["One-Time Buyer (At Risk of Never Returning)"].customers.toLocaleString()} of ${R.n_customers.toLocaleString()} customers (${((R.rfm_segments["One-Time Buyer (At Risk of Never Returning)"].customers / R.n_customers) * 100).toFixed(0)}%) have never placed a second order — this is the single largest lever for LTV improvement.`),

      sectionHeading("Recommendations"),
      bullet("Tiered shipping-time buffer: under-promise delivery dates in high-delay states (AC, RR, AM, PA, RO) rather than trying to force faster fulfillment nationally — cheaper to implement than a logistics overhaul and directly targets the 'missed expectation' mechanism.", { bold: true }),
      bullet("Targeted win-back campaign: automatically flag customers whose first order missed the estimate by >3 days and trigger a discount offer within 14 days of delivery, while the negative experience is still recoverable."),
      bullet("Category-level QA review for Garden Tools: audit top sellers in this category for packaging/quality defects independent of the shipping fix — it is under-indexed to blame on logistics alone."),

      sectionHeading("Projected Impact"),
      new Paragraph({
        spacing: { after: 60 },
        children: [new TextRun({
          text: `Cutting missed-delivery incidents by ${R.scenario.assumed_reduction_in_missed_pct}% (achievable via the shipping-buffer fix, applied to the ${R.scenario.orders_missed_gt_3days.toLocaleString()} first orders that currently miss estimate by >3 days) is projected to convert ~${R.scenario.incremental_repeat_customers.toLocaleString()} additional customers into repeat buyers, worth approximately ${money(R.scenario.estimated_annual_revenue_lift)} in incremental annual revenue at current LTV (${money(R.scenario.implied_ltv_repeat_customer)}/repeat customer). This is a conservative, mechanically-derived estimate — it does not include the compounding effect of the win-back campaign or category QA fixes above.`,
          size: 21, color: "222222",
        })],
      }),

      new Paragraph({
        spacing: { before: 200 },
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: "D9DEE7", space: 8 } },
        children: [new TextRun({
          text: "Methodology: cohort retention analysis, RFM segmentation, and a logistic regression / Random Forest churn model (AUC ≈ 0.62 — a directional, not high-precision, signal) built on order, item, review, and customer tables joined at the order-item grain. Full pipeline and dashboard available on request.",
          size: 15, italics: true, color: GREY,
        })],
      }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/olist_project/outputs/Olist_Executive_Summary.docx", buf);
  console.log("Written.");
});