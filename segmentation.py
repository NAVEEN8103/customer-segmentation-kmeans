"""Interactive customer-segmentation app.

Run with: streamlit run segmentation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "customer_segmentation_kmeans_model_v2.pkl"
SCALER_PATH = PROJECT_DIR / "customer_segmentation_scaler_v2.pkl"
METADATA_PATH = PROJECT_DIR / "segment_metadata.json"

st.set_page_config(page_title="Customer segmentation", page_icon=":material/groups:", layout="wide")


@st.cache_resource
def load_artifacts():
    if not all(path.exists() for path in (MODEL_PATH, SCALER_PATH, METADATA_PATH)):
        raise FileNotFoundError("Model files are missing. Run `python train_model.py` first.")
    return (
        joblib.load(MODEL_PATH),
        joblib.load(SCALER_PATH),
        json.loads(METADATA_PATH.read_text(encoding="utf-8")),
    )


def show_range_warnings(customer: pd.DataFrame, ranges: dict[str, dict[str, float]]) -> None:
    outside = []
    for feature, bounds in ranges.items():
        value = customer.iloc[0][feature]
        if value < bounds["min"] or value > bounds["max"]:
            outside.append(f"{feature}: {value:,.0f} (training range {bounds['min']:,.0f}–{bounds['max']:,.0f})")
    if outside:
        st.warning("Some values are outside the training range. The segment is an extrapolation: " + "; ".join(outside))


try:
    kmeans, scaler, metadata = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

features = metadata["features"]
segments = {segment["cluster_id"]: segment for segment in metadata["segments"]}

st.title("Customer segmentation studio", icon=":material/groups:")
st.caption("Classify a customer into an interpretable K-Means segment using spending, channel behavior, and recency.")
with st.container(horizontal=True):
    st.metric("Training records", f"{metadata['data_quality']['model_rows']:,}", border=True)
    st.metric("Marketing segments", metadata["n_clusters"], border=True)
    st.metric("Silhouette score (k=4)", f"{metadata['silhouette_scores']['4']:.3f}", border=True)

with st.expander("How to use this tool", expanded=False):
    st.write(
        "Enter values measured over the same customer-history period as the source dataset. "
        "The result is a behavioral segment, not a purchase prediction or a measure of customer value."
    )

with st.form("customer_inputs"):
    left, right = st.columns(2)
    with left:
        income = st.number_input("Annual income", min_value=0.0, value=50_000.0, step=1_000.0, format="%.0f")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=40, step=1)
        total_spending = st.number_input("Total spending", min_value=0.0, value=500.0, step=25.0, format="%.0f")
        recency = st.number_input("Days since last purchase", min_value=0, max_value=365, value=30, step=1)
    with right:
        num_web_purchases = st.number_input("Web purchases", min_value=0, max_value=100, value=4, step=1)
        num_store_purchases = st.number_input("Store purchases", min_value=0, max_value=100, value=5, step=1)
        num_web_visits = st.number_input("Web visits per month", min_value=0, max_value=100, value=5, step=1)
    submitted = st.form_submit_button("Identify segment", type="primary")

if submitted:
    customer = pd.DataFrame(
        [[income, age, recency, total_spending, num_web_purchases, num_store_purchases, num_web_visits]],
        columns=features,
    )
    show_range_warnings(customer, metadata["feature_ranges"])
    scaled_customer = scaler.transform(customer)
    cluster_id = int(kmeans.predict(scaled_customer)[0])
    distance = float(kmeans.transform(scaled_customer).min(axis=1)[0])
    segment = segments[cluster_id]

    st.subheader(segment["segment_name"])
    st.write(segment["description"])
    st.info(f"Recommended action: {segment['recommended_action']}")

    result_col, fit_col, share_col = st.columns(3)
    result_col.metric("Cluster ID", cluster_id, border=True)
    fit_col.metric("Distance to segment center", f"{distance:.2f}", border=True)
    share_col.metric("Segment share in training data", f"{segment['customer_share']:.1%}", border=True)

    if distance > metadata["distance_warning_threshold"]:
        st.warning("This customer is unlike most training examples. Treat the segment as directional and review the inputs.")
    else:
        st.success("This customer's profile is within the typical distance range of the training data.")

    profile_metrics = ["Income", "Total_Spending", "Recency", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth"]
    comparison = pd.DataFrame(
        {"Customer": customer.iloc[0][profile_metrics], "Typical segment customer": [segment[item] for item in profile_metrics]}
    )
    st.subheader("Customer vs. typical segment profile")
    st.bar_chart(comparison)

st.caption(
    f"Model: {metadata['model_name']} · {metadata['n_clusters']} segments · "
    f"{metadata['data_quality']['model_rows']:,} cleaned training records · random state {metadata['random_state']}"
)
