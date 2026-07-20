# Customer Segmentation Using K-Means Clustering

An end-to-end unsupervised machine learning project that groups customers into meaningful segments based on purchasing behavior, demographics, and engagement.

The project includes a complete machine learning pipeline and an interactive Streamlit application for predicting the segment of a new customer.

## Live Demo

[Open Customer Segmentation Studio](https://customer-segmentation-kmeans-gji8fbbvc6tnsfv3nmwaeh5.streamlit.app)

## GitHub Repository

[View Source Code](https://github.com/NAVEEN8103/customer-segmentation-kmeans)

---

## Project Overview

Businesses have customers with different purchasing patterns, spending habits, and levels of engagement.

This project uses K-Means clustering to divide customers into groups with similar characteristics. These groups can help businesses understand customer behavior and support targeted marketing strategies.

The application allows users to enter customer information and receive the predicted customer segment.

---

## Objectives

- Analyze customer purchasing behavior.
- Clean and preprocess customer data.
- Create useful features for segmentation.
- Apply feature scaling.
- Train a K-Means clustering model.
- Evaluate clustering quality using the silhouette score.
- Generate interpretable customer segments.
- Build an interactive Streamlit application.
- Deploy the application online.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Streamlit
- Git
- GitHub

---

## Machine Learning Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Feature Scaling
      ↓
K-Means Clustering
      ↓
Cluster Evaluation
      ↓
Segment Interpretation
      ↓
Model Saving
      ↓
Streamlit Deployment
```

---

## Dataset

The project uses customer-level data containing information related to:

- Customer demographics
- Annual income
- Purchasing behavior
- Web purchases
- Store purchases
- Website visits
- Total spending
- Customer engagement

The dataset is stored in:

```text
customer_segmentation.csv
```

---

## Features

The application provides:

- Customer input form
- Customer segment prediction
- Annual income analysis
- Age-based analysis
- Purchase behavior analysis
- Web purchase information
- Store purchase information
- Website visit information
- Segment metadata
- Interactive Streamlit interface

---

## Model Details

The project uses the K-Means clustering algorithm.

K-Means groups customers into clusters by minimizing the distance between each customer and the center of its assigned cluster.

The model uses four clusters:

```text
Number of clusters: 4
Training records: 2,212
Silhouette score: 0.193
```

The silhouette score is used to evaluate how well-separated the clusters are.

The current score indicates that the customer groups have limited separation. Further feature engineering, outlier handling, and testing different values of `k` could improve the clustering quality.

---

## Project Structure

```text
customer-segmentation-kmeans/
│
├── Analysis_model.ipynb
├── train_model.py
├── segmentation.py
├── build_notebook.py
│
├── customer_segmentation.csv
├── segment_profiles.csv
├── segment_metadata.json
│
├── customer_segmentation_kmeans_model_v2.pkl
├── customer_segmentation_scaler_v2.pkl
│
├── requirements.txt
├── .gitignore
├── README.md
│
└── .streamlit/
    └── config.toml
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NAVEEN8103/customer-segmentation-kmeans.git
```

### 2. Open the project directory

```bash
cd customer-segmentation-kmeans
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run segmentation.py
```

The application will open in your browser.

---

## Train the Model

To retrain the clustering model:

```bash
python train_model.py
```

The training script performs:

- Data loading
- Data cleaning
- Feature engineering
- Feature scaling
- K-Means training
- Silhouette score calculation
- Segment profile generation
- Model and scaler saving

---

## Main Files

### `train_model.py`

Contains the main machine learning pipeline, including preprocessing, feature engineering, clustering, evaluation, and model saving.

### `Analysis_model.ipynb`

Contains exploratory data analysis, visualizations, model experiments, and clustering analysis.

### `segmentation.py`

Contains the Streamlit application used to enter new customer information and display the predicted customer segment.

### `segment_profiles.csv`

Contains summary information about the generated customer segments.

### `segment_metadata.json`

Contains metadata used by the Streamlit application to describe the segments.

### `requirements.txt`

Contains the Python dependencies required to run the project.

---

## Results

The deployed application currently displays:

| Metric | Value |
|---|---:|
| Training records | 2,212 |
| Number of clusters | 4 |
| Silhouette score | 0.193 |

The application successfully loads the trained model and provides customer segment predictions through an interactive web interface.

---

## Limitations

- The current silhouette score indicates limited cluster separation.
- K-Means requires the number of clusters to be selected in advance.
- Results depend on the selected features and scaling method.
- Customer segments are descriptive groups and should not be treated as absolute customer categories.
- The model is trained on the available dataset and may not represent every customer population.

---

## Future Improvements

- Test different values of `k`.
- Compare K-Means with other clustering algorithms.
- Improve feature selection and feature engineering.
- Handle outliers more carefully.
- Add PCA-based cluster visualization.
- Add customer segment comparison charts.
- Improve segment naming and business interpretation.
- Add batch customer prediction.
- Add model monitoring and retraining functionality.

---

## Author

**Naveen Kumar Tiwari**

- GitHub: [NAVEEN8103](https://github.com/NAVEEN8103)
- Project: [Customer Segmentation Using K-Means](https://github.com/NAVEEN8103/customer-segmentation-kmeans)
- Live Demo: [Customer Segmentation Studio](https://customer-segmentation-kmeans-gji8fbbvc6tnsfv3nmwaeh5.streamlit.app)