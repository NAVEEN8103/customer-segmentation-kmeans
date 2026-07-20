# Customer Segmentation with K-Means

An end-to-end unsupervised-learning project that groups retail customers into actionable segments using behavior, spending, and recency signals. The project includes a reproducible training pipeline and an interactive Streamlit scoring app.

## Business question

How can a marketing team group customers into distinct behavioral segments and tailor retention, loyalty, and conversion actions to each group?

## What this project demonstrates

- Data-quality checks for missing income, implausible ages, and an extreme income record
- Feature engineering for customer age and total product spending
- Standardization before distance-based clustering
- Reproducible K-Means training (`random_state=42`, `n_init=20`)
- Cluster-quality comparison with silhouette scores for `k=2` through `k=8`
- Business-friendly segment naming and recommended marketing actions
- An interactive Streamlit app with range checks and out-of-distribution warnings

## Segments

The app loads segment names from `segment_metadata.json` after training. This avoids treating K-Means' arbitrary numeric labels as business meaning.

| Segment | Typical use |
| --- | --- |
| Elite / VIP | Protect and grow the highest-value relationships. |
| Premium Active | Cross-sell and strengthen loyalty across channels. |
| At-Risk Browsers | Run targeted win-back and conversion campaigns. |
| Low-Value Recent | Nurture with relevant, accessible offers. |

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run segmentation.py
```

## Project files

- `customer_segmentation.csv` — source customer data
- `train_model.py` — cleaning, feature engineering, model training, validation, and artifact export
- `Analysis_model.ipynb` — exploratory and modeling notebook
- `segmentation.py` — interactive Streamlit app
- `segment_metadata.json` and `segment_profiles.csv` — generated, interpretable model outputs

## Model-selection note

The pipeline calculates silhouette scores for `k=2` through `k=8`. On this source dataset, `k=2` has the highest score (0.3286); the checked-in `k=4` model (0.1929) is an intentional business-actionability configuration that separates the high-value, at-risk, and low-value audiences into distinct marketing playbooks. In a production setting, validate this choice with campaign outcomes and stakeholder review.

## Important limitation

This is an exploratory segmentation model, not a causal model or a purchase-propensity predictor. A submitted customer should be measured over the same period and definitions as the training data; the app flags values outside the observed training range.
