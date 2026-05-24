# 💼 Finance Risk Analytics Platform

> A full-stack machine learning application that helps financial institutions assess customer credit risk, predict loan defaults, estimate customer lifetime value, and segment customers into behavioral profiles — all through an interactive web dashboard.

---

## 🧠 The Problem It Solves

Banks and financial institutions process thousands of loan applications every year. Manually assessing each customer's risk is slow, inconsistent, and expensive. This platform automates that process by running a customer's financial profile through **6 machine learning models** simultaneously and presenting the results in a clean, interactive dashboard.

---

## 🔍 What It Can Do

### 🔮 Predict Loan Default
Enter a customer's financial details and instantly get default predictions from 4 different models side by side. Each model returns a probability score — above 50% means the customer is likely to default on their loan.

### 💰 Estimate Customer Lifetime Value
Uses Linear Regression to predict how much revenue a customer will generate over their relationship with the institution. Helps prioritize which customers are worth acquiring or retaining.

### 🗂️ Customer Segmentation
Groups all customers in the database into 4 behavioral clusters using K-Means — Low Risk, Medium Risk, High Risk, and Very High Risk. Visualized as interactive scatter plots so patterns are immediately visible.

### 📈 Model Comparison
Compare the performance of all 4 classifiers (Accuracy, F1 Score, ROC-AUC) in one view. Useful for understanding which model to trust most for a given scenario.

### 🗄️ Customer Database
Full CRUD interface to browse, search, and manage all customers and their prediction history stored in PostgreSQL.

---

## 🤖 Machine Learning Models

| Model | Task | Why It's Used |
|---|---|---|
| Logistic Regression | Default prediction | Simple baseline, highly interpretable |
| Random Forest | Default prediction | Handles non-linear patterns, robust to outliers |
| XGBoost | Default prediction | Best performance on tabular financial data |
| ANN (MLPClassifier) | Default prediction | Learns complex feature interactions |
| Linear Regression | Customer Lifetime Value | Predicts a continuous dollar value |
| K-Means | Customer Segmentation | Groups customers by behavioral similarity |

All 6 models are trained on the same 7 features: **age, income, debt ratio, monthly expenses, late payments, number of loans, and dependents.**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────┐
│         Streamlit Frontend          │
│  Dashboard · Predict · CLV · Seg    │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│            predict.py               │
│   Loads .pkl models · Runs inference│
└────────────────┬────────────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
┌──────▼──────┐    ┌────────▼────────┐
│  models/    │    │   database.py   │
│  .pkl files │    │   PostgreSQL    │
│  (trained)  │    │  customers      │
└─────────────┘    │  predictions    │
                   │  model_metrics  │
                   └─────────────────┘
```

---

## 📁 Project Structure

```
finance-risk-platform/
├── app.py          # Streamlit UI — all pages and navigation
├── models.py       # Train all 6 ML models and save as .pkl
├── predict.py      # Load saved models and run inference
├── database.py     # PostgreSQL connection, schema, CRUD operations
└── models/         # Saved .pkl model files (generated after training)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- PostgreSQL installed and running
- Git

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/finance-risk-platform.git
cd finance-risk-platform
```

### 2. Install dependencies
```bash
pip install {required library}
```

### 3. Configure the database
Create a database named `finance_risk` in PostgreSQL, then copy `.env.example` to `.env` and fill in your credentials:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finance_risk
DB_USER=postgres
DB_PASSWORD=your_password
```

### 4. Create tables
```bash
python database.py
```

### 5. Train all models
```bash
python models.py
```
This trains all 6 algorithms and saves them as `.pkl` files in the `models/` folder. You'll see a metrics table printed when done.

### 6. Launch the app
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📊 Model Performance (Synthetic Dataset)

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 94.0% | 0.927 | 0.974 |
| Random Forest | 91.5% | 0.894 | 0.953 |
| XGBoost | 89.0% | 0.868 | 0.944 |
| ANN (MLPClassifier) | 89.0% | 0.866 | 0.953 |

> Note: These results are from the built-in synthetic dataset. Swap in a real dataset(from kaggle or other websites) for production-level results.

---

## 🔄 Using a Real Dataset

The project ships with a synthetic dataset for demonstration. To use real data, replace `generate_dataset()` in `models.py` with:

```python
df = pd.read_csv("data/your_dataset.csv")
```

Make sure your CSV has these columns: `age`, `income`, `debt_ratio`, `monthly_expenses`, `late_payments`, `num_loans`, `dependents`, `default`

Recommended dataset: **Give Me Some Credit** from Kaggle — see `models.py` for the exact column mapping code.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Models | scikit-learn, XGBoost |
| Database | PostgreSQL + psycopg2 + SQLAlchemy |
| Data Processing | pandas, NumPy |
| Visualization | Plotly |
| Model Persistence | joblib |

---

## 🎓 Academic Context

Built as a semester project for **AI221 — Machine Learning** demonstrating:
- Supervised learning (classification + regression)
- Unsupervised learning (clustering)
- Model evaluation and comparison
- Database integration with a live ML application
- End-to-end deployment with a web interface