# Bank Customer Segmentation & Satisfaction (NLP + Clustering)

Unsupervised analysis of 1,000 customers of **Chase, Bank of America and Wells Fargo**:
1. **Customer segmentation** — homogeneous groups with a tailored financial-product offer, and the sentiment of each group.
2. **Satisfaction analysis** — satisfaction level from customer opinions, the main reason behind it, and how it varies across customers.

## Key results

| | |
|---|---|
| Average satisfaction | **4.17★** (60% very satisfied · 19% passive · 21% dissatisfied) |
| #1 driver of delight *and* frustration | **Digital & mobile** (app reliability, crashes during payment) |
| Satisfaction vs. customer profile or bank | No significant relationship (all p > 0.10) |
| Segments | **5** (K-Means): Young professionals · Established homeowners · Mature multi-product · Highly leveraged earners · Under-engaged customers |
| Deliverable | Saved pipeline that assigns any new customer to a segment and returns its offer |

![Reasons by satisfaction level](reports/figures/07_reasons_by_level.png)
![Segment profiles](reports/figures/11_segment_profiles.png)

## Repository structure

```
bank-customer-satisfaction/
├── README.md
├── requirements.txt
├── notebooks/
│   └── CustomersSatisfaction_NLP.ipynb   # full analysis (Modules 0–15)
│   └── segment_assigner.py               # assign new customers to a segment
├── data/
│   ├── raw/                              # bank_customers_satisfaction.xlsx (never modified)
├── reports/
│   ├── Executive_Summary_Bank_Customers.docx
│   ├── cleaning_log.csv · feature_dictionary.csv · eda_findings.xlsx
│   ├── segment_offers.csv · segmentation_model_comparison.csv · dbscan_grid.csv
│  
└──  Figures for Data Visualization/                          # every chart, prefixed by module (02_…, 03_…)
```

## Notebook modules

| # | Module | Rubric item |
|---|---|---|
| 1–2 | Loading, diagnosis and cleaning (with audit log) | Data cleaning |
| 3 | Exploratory analysis and preliminary findings | Exploration |
| 4 | Encoding, new variables, skew correction | Transformation |
| 5 | Text cleaning, topic extraction, business taxonomy | Message processing |
| 6 | Sentiment models (VADER, RoBERTa, 5-star) and satisfaction level | Satisfaction levels |
| 7 | Topic embeddings + hierarchical clustering → main reason | Main reason |
| 8 | Satisfaction vs. customer characteristics | Characterization |
| 9–11 | Feature selection, K-Means / hierarchical / DBSCAN comparison, selected model and segments | Model development |
| 12 | Saved pipeline + assignment module, tested on new customers | Saved model |
| 13–14 | Executive reports (segmentation, satisfaction) | Executive reports |

## How to run

1. Open `notebooks/CustomersSatisfaction_NLP.ipynb` in Google Colab (a T4 GPU speeds up the transformer models).
2. Place `bank_customers_satisfaction.xlsx` in your project folder and set `PROJECT_DIR` in cell 1.1.
3. Run all cells. Outputs are written to `outputs/{data, models, figures, reports}`.

## Assign new customers

```python
import pandas as pd
from src.segment_assigner import assign_segment

new = pd.DataFrame([{"client_age": 26, "customer_since": 2023, "bank_account": "yes",
                     "credit_card": "no", "mortgage": "no", "total_number_of_products": 1,
                     "customer_income": 120000, "customer_debt": 90000,
                     "use_of_digital_channels": "high", "use_of_branches": "low"}])
assign_segment(new, "models/segmentation_kmeans_v1.joblib")
```

## Methods at a glance
- **Cleaning:** type scan of every cell, business-rule checks, IQR / z-score / Isolation Forest outliers; nothing deleted, everything flagged and logged.
- **NLP:** rule-based topic extraction; pre-trained sentiment models used without training (validated against a 3-template reference); `all-MiniLM-L6-v2` embeddings clustered with agglomerative clustering (cosine).
- **Segmentation:** 10 standardized features; K-Means, hierarchical clustering (Ward / average / complete linkage) and DBSCAN compared on silhouette, Calinski-Harabasz, Davies-Bouldin, noise share and bootstrap stability.

## Limitations
Customer variables are uniform and independent and opinions follow three templates, which points to a synthetic dataset. Segments are a practical partition (silhouette ≈ 0.10) rather than natural clusters.

## Author
Juan Esteban Londoño-Marín
