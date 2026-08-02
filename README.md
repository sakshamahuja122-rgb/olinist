# Olist Retention Diagnostic — A Consulting-Style Analytics Case Study

This repository contains the Olist retention diagnostic case study: cohort retention, RFM segmentation, churn diagnostics (logistic regression + Random Forest), and a small Word executive summary generator.

Summary (excerpt):

- First-order churn rate: ~88% (example synthetic result)
- Order defect rate (1–2★): ~20%
- Delivery >3 days late churn rate ~96% vs ~86% on-time — delivery reliability is the strongest churn driver

Repository layout (local project expected under `olist_project_package`):

scripts/
  01_generate_data.py         # builds the synthetic dataset (or use real Kaggle CSVs)
  02_analysis.py             # full pipeline: joins → cohort/RFM → churn model → $ impact
  03_build_exec_summary.js   # builds a 1-page Word executive summary
data/                        # CSVs: orders.csv, order_items.csv, order_reviews.csv, customers.csv
outputs/                     # generated results.json, charts, dashboard, and a docx summary

Requirements

- Python 3.8+ with: pandas, numpy, scikit-learn, matplotlib
- Node.js (16+) with the `docx` package (used by the exec summary generator)

Quick setup (Windows PowerShell)

1) Install Python dependencies:

   python -m pip install --upgrade pip
   pip install -r requirements.txt

2) Install Node dependencies (in the repo root):

   npm install

Running the pipeline (Windows)

Note: the packaged scripts use absolute Unix-style paths (/home/claude/olist_project/...). On Windows these need to be updated to point to the local project folder. Two options:

A) Edit scripts to set DATA and OUT variables to the absolute Windows paths in your environment. Replace the top-level DATA and OUT values in `scripts/01_generate_data.py`, `scripts/02_analysis.py`, and `scripts/03_build_exec_summary.js` with the Windows paths below (example for the provided layout):

   DATA = "C:\\Users\\saksh\\Downloads\\olist_retention_project\\olist_project_package\\data"
   OUT  = "C:\\Users\\saksh\\Downloads\\olist_retention_project\\olist_project_package\\outputs"

B) Or move the `olist_project_package` folder to a Unix-like environment (WSL) and run there.

After adjusting paths, create the outputs folder if missing and run:

   python .\olist_project_package\scripts\01_generate_data.py
   python .\olist_project_package\scripts\02_analysis.py
   node .\olist_project_package\scripts\03_build_exec_summary.js

What this push includes

- This README.md (instructions and summary)
- requirements.txt (Python dependencies)
- package.json (Node dependency for doc generation)
- .gitignore (ignore outputs and caches)

Notes & next steps

- The repo currently contains the project manifest and instructions. If you want the full dataset and generated outputs pushed as well, confirm and those files can be uploaded (note: some outputs are binary — PNG/DOCX — and will increase repo size).
- If you want the assistant to also open a PR, commit the repo to a specific organization, or modify the scripts to be Windows-friendly (use relative paths or environment variables), say which option.

If anything should be changed (repo name, make scripts read DATA/OUT from env vars, or push the full files), reply and the assistant will proceed.
