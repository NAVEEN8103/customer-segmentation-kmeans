"""Train and export a reproducible customer-segmentation model.

Run from this directory:
    python train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "customer_segmentation.csv"
MODEL_PATH = PROJECT_DIR / "customer_segmentation_kmeans_model_v2.pkl"
SCALER_PATH = PROJECT_DIR / "customer_segmentation_scaler_v2.pkl"
METADATA_PATH = PROJECT_DIR / "segment_metadata.json"
PROFILES_PATH = PROJECT_DIR / "segment_profiles.csv"

RANDOM_STATE = 42
N_CLUSTERS = 4
REFERENCE_YEAR = 2015  # One year after the latest customer-enrollment date in the source data.

FEATURES = [
    "Income", "Age", "Recency", "Total_Spending", "NumWebPurchases",
    "NumStorePurchases", "NumWebVisitsMonth",
]
SPEND_COLUMNS = [
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds",
]


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean source data and derive the modeling features."""
    records = {"raw_rows": len(data)}
    clean = data.dropna(subset=["Income"]).copy()
    records["dropped_missing_income"] = records["raw_rows"] - len(clean)

    clean["Age"] = REFERENCE_YEAR - clean["Year_Birth"]
    clean["Total_Spending"] = clean[SPEND_COLUMNS].sum(axis=1)
    valid_age = clean["Age"].between(18, 100)
    valid_income = clean["Income"].between(0, 250_000)
    records["dropped_invalid_age_or_income"] = int((~(valid_age & valid_income)).sum())
    clean = clean.loc[valid_age & valid_income].copy()
    records["model_rows"] = len(clean)
    if clean.empty:
        raise ValueError("No records remain after data-quality checks.")
    return clean, records


def assign_segment_names(profiles: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary K-Means IDs to stable, business-friendly segment names."""
    result = profiles.copy()
    ranked_spend = result["Total_Spending"].sort_values(ascending=False).index.tolist()
    vip_cluster, premium_cluster = ranked_spend[:2]
    remaining = [cluster for cluster in result.index if cluster not in {vip_cluster, premium_cluster}]
    at_risk_cluster = result.loc[remaining, "Recency"].idxmax()
    recent_low_value_cluster = next(cluster for cluster in remaining if cluster != at_risk_cluster)

    definitions = {
        vip_cluster: ("Elite / VIP", "Highest-value customers: strongest income and total spending with low browsing activity.", "Offer exclusive rewards, early access, and concierge-style retention experiences."),
        premium_cluster: ("Premium Active", "High-value customers who purchase frequently across web and store channels.", "Use cross-sell bundles and loyalty offers to increase lifetime value."),
        at_risk_cluster: ("At-Risk Browsers", "Lower-value customers with the longest time since purchase and comparatively high web visits.", "Use timely win-back messages, targeted incentives, and friction-reduction experiments."),
        recent_low_value_cluster: ("Low-Value Recent", "Lower-income, lower-spending customers who have purchased relatively recently.", "Nurture with entry-level offers and relevant product recommendations."),
    }
    result["segment_name"] = [definitions[index][0] for index in result.index]
    result["description"] = [definitions[index][1] for index in result.index]
    result["recommended_action"] = [definitions[index][2] for index in result.index]
    return result


def main() -> None:
    raw_data = pd.read_csv(DATA_PATH)
    clean_data, quality_summary = prepare_data(raw_data)
    feature_data = clean_data[FEATURES].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    clean_data["cluster_id"] = model.fit_predict(scaled_features)

    profiles = clean_data.groupby("cluster_id")[FEATURES].mean()
    profiles["customer_count"] = clean_data["cluster_id"].value_counts().sort_index()
    profiles["customer_share"] = profiles["customer_count"] / len(clean_data)
    profiles = assign_segment_names(profiles).reset_index()

    distances = model.transform(scaled_features).min(axis=1)
    feature_ranges = {feature: {"min": float(feature_data[feature].min()), "max": float(feature_data[feature].max())} for feature in FEATURES}
    silhouette_scores = {str(k): round(float(silhouette_score(scaled_features, KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(scaled_features))), 4) for k in range(2, 9)}
    metadata = {
        "model_name": "K-Means customer segmentation", "model_version": "1.0",
        "random_state": RANDOM_STATE, "n_clusters": N_CLUSTERS,
        "reference_year_for_age": REFERENCE_YEAR, "features": FEATURES,
        "data_quality": quality_summary, "silhouette_scores": silhouette_scores,
        "distance_warning_threshold": float(np.percentile(distances, 95)),
        "feature_ranges": feature_ranges, "segments": profiles.to_dict(orient="records"),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    profiles.to_csv(PROFILES_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Training complete")
    print(f"Model rows: {quality_summary['model_rows']}")
    print("Silhouette scores:", silhouette_scores)
    print(profiles[["cluster_id", "segment_name", "customer_count", "customer_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
