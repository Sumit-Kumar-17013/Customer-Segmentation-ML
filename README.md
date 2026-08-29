<div align="center">

# 🎯 CustomerIQ — AI-Powered Customer Segmentation
### *End-to-End Unsupervised Machine Learning Pipeline & Real-Time Intelligence Engine*

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>Transforming raw demographic, transactional, and engagement signals into actionable, high-impact marketing strategies.</b>
</p>

[Explore Pipeline](#-machine-learning-approach) • [Cluster Profiles](#-final-customer-segments) • [FastAPI Reference](#-fastapi-backend-service) • [Render Deployment](#-deployment-architecture) • [Getting Started](#-local-setup--quickstart)

</div>

---

## 📖 Executive Summary

**CustomerIQ** is a production-grade, end-to-end customer segmentation platform. Rather than applying generic blanket marketing campaigns, this system extracts latent behavioral patterns from high-dimensional customer data—enabling personalized retention, targeted discounts, and concierge experiences.

The core machine learning engine couples **non-linear skew handling ($\log(1+x)$)**, **standardization**, **PCA dimensionality reduction (retaining 90% cumulative variance)**, and **K-Means clustering** into a single, unified Scikit-Learn pipeline deployed behind a high-performance **FastAPI** inference microservice.

---

## 🌟 Key Highlights & Engineering Features

- 🔬 **Rigorous Exploratory Data Analysis (EDA)**: IQR-based outlier detection, feature skewness auditing, and duplicate validation.
- 📐 **Skew-Corrected Spend Normalization**: Integrated `log1p(total_spend)` step inside the pipeline to eliminate training-serving skew.
- 🧪 **Multi-Algorithm Evaluation**: Benchmarked **K-Means**, **Agglomerative Hierarchical Clustering**, and **Gaussian Mixture Models (GMM)** against Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.
- 🎯 **Business-Optimized $K$-Value ($K=2$)**: Avoided an unviable ultra-small cluster ($0.89\%$) observed at $K=3$ to deliver high-confidence, actionable business segments.
- ⚡ **Asynchronous FastAPI Engine**: Complete with `/Health` diagnostics, uptime tracking, and `/Prediction` inference.
- 💻 **High-Density Web UI**: Dark glassmorphic workspace featuring live latency tracking, sample archetype presets, and interactive PCA scatter mapping.

---

## 🏗 Machine Learning Approach

The end-to-end data lifecycle follows a strict transformation flow designed to guarantee zero data leakage:

```text
       ┌─────────────────────────────────────────────────────────┐
       │             Raw Multi-Dimensional Customer Data         │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │   Data Cleaning & Exploratory Data Analysis (EDA)       │
       │   (IQR Outlier Detection, Null Checks, Type Casts)      │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │       Feature Engineering & Encoding Categoricals       │
       │       (Education_Encoded, Living_With_Encoded)          │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │       Pipeline Transformation Layer (ColumnTransformer) │
       │       • total_spend ───► log1p(total_spend)             │
       │       • Numerical Features ───► StandardScaler (μ=0, σ=1)│
       │       • Variance Reduction ───► PCA (90% Variance)      │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │      K-Means Clustering Engine (K=2 Centroids)          │
       └────────────────────────────┬────────────────────────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
       ┌─────────────────────────────┐ ┌─────────────────────────────┐
       │   Cluster 0: Browsing /     │ │   Cluster 1: High-Value     │
       │   Value-Conscious Customers │ │   Core Revenue Drivers      │
       └──────────────┬──────────────┘ └──────────────┬──────────────┘
                      │                               │
                      ▼                               ▼
       ┌─────────────────────────────┐ ┌─────────────────────────────┐
       │  Action: Conversion Offers, │ │  Action: VIP Concierge,     │
       │  Bundles, Cart Incentives   │ │  Exclusive Access, Upgrades │
       └─────────────────────────────┘ └─────────────────────────────┘
```

---

## 📊 Features Used in Final Production Model

The final segmentation model evaluates **12 refined demographic and behavioral features**:

| # | Feature Name | Data Type | Constraint / Domain | Description |
|---|---|---|---|---|
| 1 | `Age` | `float` | $18 \le \text{Age} \le 100$ | Customer age in years |
| 2 | `Income` | `float` | $\ge 0$ | Annual household income in USD |
| 3 | `Recency` | `float` | $\ge 0$ | Days elapsed since last recorded purchase |
| 4 | `Customer_Tenure_Days` | `float` | $\ge 0$ | Total days associated with the brand |
| 5 | `total_spend` | `float` | $\ge 0$ | **Original spend in USD** *(pipeline auto-applies $\log(1+x)$)* |
| 6 | `total_purchase` | `float` | $\ge 0$ | Total number of purchase transactions |
| 7 | `total_campaigns` | `float` | $\ge 0$ | Total marketing campaigns accepted |
| 8 | `children` | `float` | $\ge 0$ | Number of dependent children in household |
| 9 | `family_size` | `float` | $\ge 1$ | Total members in household |
| 10 | `Education_Encoded` | `float` | `0`, `1`, `2` | `0`: Undergraduate/Basic, `1`: Graduate, `2`: Postgraduate/PhD |
| 11 | `Living_With_Encoded` | `float` | `0`, `1` | `0`: Alone / Single, `1`: Partner / Cohabitating |
| 12 | `NumWebVisitsMonth` | `float` | $\ge 0$ | Total website visits logged in previous 30 days |

> **Note:** The `Complain` attribute was excluded from the final clustering matrix to maintain clear separation of core purchasing behavior.

---

## 🔬 Algorithm Benchmarking & $K$-Value Selection

Multiple unsupervised algorithms were benchmarked across dimensionality states:

| Algorithm | Dimensionality Reduction | Evaluated Clusters ($K$) | Primary Evaluation Metric |
|---|---|---|---|
| **K-Means** | Raw & PCA (90% Var) | $K \in [2, 10]$ | Silhouette, Davies-Bouldin, Inertia |
| **Agglomerative Hierarchical** | Raw & PCA (90% Var) | $n \in [2, 10]$ | Silhouette, Calinski-Harabasz |
| **Gaussian Mixture Models (GMM)**| Raw & PCA (90% Var) | $n \in [2, 10]$ | BIC / AIC, Silhouette Score |

### Why $K = 2$?
* **The $K=3$ Paradox**: In PCA + K-Means trials, $K=3$ reached a mathematical Silhouette score of $\approx 0.2237$. However, the 3rd cluster isolated an extreme micro-group comprising just **$0.89\%$ of the entire dataset**.
* **Business Viability**: Selecting **$K=2$** created balanced, operationally meaningful segments with distinct purchasing power and browsing intent.

---

## 👥 Final Customer Segments & Business Strategies

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SEGMENT PROFILE COMPARISON                                │
├────────────────────────────────┬───────────────────────────┬───────────────────────────┤
│ Metric / Behavioral Feature    │ Cluster 0 (Browsing)      │ Cluster 1 (High-Value)    │
├────────────────────────────────┼───────────────────────────┼───────────────────────────┤
│ Average Age                    │ 42.9 Years                │ 47.5 Years                │
│ Average Annual Income          │ $36,215.00                │ $69,071.40                │
│ Average Recency                │ 49.2 Days                 │ 49.0 Days                 │
│ Average Customer Tenure        │ 335.7 Days                │ 373.5 Days                │
│ Average Total Spend            │ $141.20                   │ $1,111.60                 │
│ Average Purchase Count         │ 9.3 Transactions          │ 20.9 Transactions         │
│ Campaigns Accepted             │ 0.1 Campaigns             │ 0.5 Campaigns             │
│ Household Children             │ 1.3 Dependents            │ 0.6 Dependents            │
│ Average Family Size            │ 3.0 Members               │ 2.2 Members               │
│ Monthly Web Visits             │ 6.5 Visits / Month        │ 3.7 Visits / Month        │
└────────────────────────────────┴───────────────────────────┴───────────────────────────┘
```

### 🟡 Cluster 0 — Low-Value / Browsing Customers
- **Behavioral Diagnosis**: High interest and web presence ($6.5\text{ visits/mo}$), but lower disposable income and modest spending ($141.20).
- **Targeted Action Plan**:
  1. **Conversion Triggers**: Automated abandoned-browse emails with limited-time 10–15% discount coupons.
  2. **Starter Bundles**: Packaged introductory offerings with low friction-to-purchase barriers.
  3. **Affinity Recommendations**: Surface high-value, top-rated products based on browsing logs.

---

### 🟢 Cluster 1 — High-Value Customers
- **Behavioral Diagnosis**: Established buyers with high income ($69,071.40) and high cumulative spending ($1,111.60) across 20+ purchase transactions. Represents the primary revenue driver.
- **Targeted Action Plan**:
  1. **VIP Loyalty Concierge**: Tier-1 status rewards with dedicated customer assistance and expedited fulfillment.
  2. **Premium Cross-Selling**: Early-bird access to flagship releases and high-margin bespoke bundles.
  3. **Retention Defense**: Exclusive invitations to private webinars, product launches, and loyalty milestones.

---

## 📁 Repository Directory Structure

```
customer-segmentation/
│
├── main.py                             # FastAPI backend application & routing
├── customer_segmentation_pipeline.pkl  # Serialized scikit-learn ML pipeline
├── requirements.txt                    # Pinned Python backend dependencies
├── README.md                           # Master project documentation
│
├── index.html                          # Semantic High-Density interface & modals
├── style.css                           # High-Density theme styling & animations
├── script.js                           # Frontend controller, validation & API handler
│
└── notebook/
    └── customer_segmentation.ipynb     # Jupyter notebook (EDA, training & validation)
```

---

## ⚡ FastAPI Backend Service

The application exposes a clean RESTful interface:

### 1. `POST /Prediction` — Segment Inference
Submit raw customer attributes and receive cluster classification with business messaging.

#### Request Body
```json
{
  "Age": 45,
  "Income": 78500,
  "Recency": 28,
  "Customer_Tenure_Days": 420,
  "total_spend": 1240,
  "total_purchase": 24,
  "total_campaigns": 2,
  "children": 1,
  "family_size": 3,
  "Education_Encoded": 1,
  "Living_With_Encoded": 1,
  "NumWebVisitsMonth": 4
}
```

#### Response Body
```json
{
  "cluster": 1,
  "segment": "High-Value Customers",
  "message": "High-value customers with higher income, spending and purchase activity. Primary revenue engine."
}
```

---

### 2. `GET /Health` — Uptime & System Health
```json
{
  "status": "ok",
  "timestamp": "2026-08-29T21:15:00.000000+00:00",
  "uptime": 348.2
}
```

---

## 💻 Local Setup & Quickstart

### 1. Backend Setup (FastAPI)

```bash
# Clone the repository
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation

# Create & activate virtual environment
# Windows PowerShell:
python -m venv cenv
cenv\Scripts\Activate.ps1

# macOS / Linux:
python3 -m venv cenv
source cenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch FastAPI development server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

### 2. Frontend Setup

```bash
# Option A: Simple Python HTTP Server
python3 -m http.server 3000

# Option B: Node.js / Vite
npm install
npm run dev
```

Open `http://localhost:3000` in your web browser.

---

## 🌐 Deployment Architecture (Render)

```text
 ┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
 │     Render Static Site (Frontend)    │       │     Render Web Service (FastAPI)     │
 │                                      │       │                                      │
 │   • index.html, style.css, script.js │ ────► │   • main.py                          │
 │   • High-Density Theme & Presets     │       │   • customer_segmentation_pipe.pkl   │
 │   • Dynamic API URL Switcher         │       │   • Port: $PORT via Uvicorn          │
 └──────────────────────────────────────┘       └──────────────────────────────────────┘
```

### Deploying Backend to Render
1. Create a **New Web Service** on [Render](https://render.com/).
2. Set **Runtime** to `Python 3`.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Verify `main.py`, `customer_segmentation_pipeline.pkl`, and `requirements.txt` exist in your repository root.

### Deploying Frontend to Render
1. Create a **New Static Site** on Render.
2. Set **Publish Directory** to `.` (or `dist` if building via Vite).
3. Update line 17 in `script.js` with your live FastAPI backend URL:
   ```javascript
   const API_BASE_URL = "https://your-fastapi-backend.onrender.com";
   ```
   *(Or configure it live using the top-bar **API Status badge** inside the app).*

---

## 🛠 Tech Stack

| Category | Technologies |
|---|---|
| **Core Languages** | Python 3.10+, JavaScript (ES6+), HTML5, Modern CSS3 |
| **Data Science & ML** | Scikit-Learn, Pandas, NumPy, SciPy, Matplotlib, Seaborn |
| **Pipeline Components**| `ColumnTransformer`, `FunctionTransformer`, `StandardScaler`, `PCA`, `KMeans` |
| **API Framework** | FastAPI, Uvicorn, Pydantic |
| **Model Serialization**| Joblib / Pickle |
| **Cloud Hosting** | Render (Web Service + Static Site) |

---

## 🔮 Future Roadmap

- [ ] Automated real-time customer lifetime value (CLV) regression scoring.
- [ ] Integration of RFM (Recency, Frequency, Monetary) dynamic scorecards.
- [ ] Automated email marketing webhook triggers via SendGrid / Resend.
- [ ] Model drift detection and scheduled retraining pipelines.

---

## 👤 Author

**Sumit Kumar**  
*Machine Learning & Full-Stack Intelligence Engineer*  
- **Project**: Customer Segmentation using Machine Learning
- **Repository**: [CustomerIQ Intelligence Engine](https://github.com/)

---

<div align="center">
  <sub>Crafted with precision for enterprise customer analytics. Licensed under the MIT License.</sub>
</div>
