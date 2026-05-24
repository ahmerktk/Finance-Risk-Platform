import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import database as db
import predict as pr

st.set_page_config(
    page_title="Finance Risk Platform",
    page_icon="💼",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("💼 Finance Risk")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🔮 Predict Default",
    "💰 CLV Estimator",
    "🗂️ Segmentation",
    "📈 Model Comparison",
    "🗄️ Customer Database"
])

st.sidebar.markdown("---")
st.sidebar.caption("Powered by sklearn · XGBoost · PostgreSQL")


# Dashboard Page
if page == "📊 Dashboard":
    st.title("📊 Dashboard")

    try:
        customers = db.get_all_customers()
        metrics   = db.get_all_metrics()

        if customers.empty:
            st.info("No customers in the database yet. Add some via 'Predict Default'.")
        else:
            # KPI row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Customers", len(customers))
            col2.metric("Avg Income",      f"${customers['income'].mean():,.0f}")
            col3.metric("Avg Debt Ratio",  f"{customers['debt_ratio'].mean():.2%}")
            col4.metric("Avg Late Payments", f"{customers['late_payments'].mean():.1f}")

            st.markdown("---")

            # Charts row
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Income Distribution")
                fig = px.histogram(customers, x="income", nbins=30,
                                   color_discrete_sequence=["#4f8ef7"])
                fig.update_layout(margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.subheader("Cluster Distribution")
                if customers["cluster_label"].eq(-1).all():
                    st.info("Run segmentation to assign clusters.")
                else:
                    cluster_counts = customers["cluster_label"].value_counts().reset_index()
                    cluster_counts.columns = ["Cluster", "Count"]
                    fig2 = px.pie(cluster_counts, values="Count", names="Cluster",
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig2, use_container_width=True)

        # Model metrics table
        if not metrics.empty:
            st.markdown("---")
            st.subheader("Latest Model Metrics")
            st.dataframe(
                metrics[["model_name", "accuracy", "f1_score", "roc_auc"]]
                .rename(columns={"model_name": "Model", "accuracy": "Accuracy",
                                  "f1_score": "F1 Score", "roc_auc": "ROC-AUC"})
                .set_index("Model"),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Database error: {e}")
        st.info("Make sure PostgreSQL is running and you've run `python database.py` to create tables.")


# Predict Default Page
elif page == "🔮 Predict Default":
    st.title("🔮 Predict Credit Default")
    st.write("Enter customer details to get default predictions from all 4 models.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age              = st.number_input("Age",              min_value=18, max_value=100, value=35)
            income           = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
            debt_ratio       = st.slider("Debt Ratio",             0.0, 1.0, 0.3, 0.01)
        with col2:
            monthly_expenses = st.number_input("Monthly Expenses ($)", min_value=0.0, value=2000.0, step=100.0)
            late_payments    = st.number_input("Late Payments (count)", min_value=0, max_value=20, value=1)
        with col3:
            num_loans        = st.number_input("Number of Loans",  min_value=0, max_value=15, value=2)
            dependents       = st.number_input("Dependents",       min_value=0, max_value=10, value=1)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        input_dict = dict(age=age, income=income, debt_ratio=debt_ratio,
                          monthly_expenses=monthly_expenses, late_payments=late_payments,
                          num_loans=num_loans, dependents=dependents)
        try:
            results = pr.predict_default(input_dict)

            st.markdown("### Results")
            cols = st.columns(len(results))
            for col, r in zip(cols, results):
                color = "🔴" if r["prediction"] == "Default" else "🟢"
                col.metric(
                    label=r["model"],
                    value=f"{color} {r['prediction']}",
                    delta=f"Prob: {r['probability']:.2%}"
                )

            # Probability bar chart
            fig = px.bar(
                x=[r["model"] for r in results],
                y=[r["probability"] for r in results],
                labels={"x": "Model", "y": "Default Probability"},
                color=[r["probability"] for r in results],
                color_continuous_scale="RdYlGn_r",
                range_y=[0, 1]
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                          annotation_text="Decision threshold (0.5)")
            st.plotly_chart(fig, use_container_width=True)

            # Save to DB
            cluster_label, _ = pr.predict_cluster(input_dict)
            customer_id = db.insert_customer(**input_dict, cluster_label=cluster_label)
            for r in results:
                db.insert_prediction(customer_id, r["model"], r["prediction"], r["probability"])
            st.success(f"✅ Customer #{customer_id} saved to database.")

        except FileNotFoundError:
            st.error("Models not found. Please run `python models.py` first.")
        except Exception as e:
            st.error(f"Error: {e}")


# CLV (Customer Lifetime Value) Estimation
elif page == "💰 CLV Estimator":
    st.title("💰 Customer Lifetime Value Estimator")
    st.write("Uses **Linear Regression** to predict how much value a customer will generate.")

    with st.form("clv_form"):
        col1, col2 = st.columns(2)
        with col1:
            age              = st.number_input("Age",               min_value=18, max_value=100, value=35)
            income           = st.number_input("Annual Income ($)",  min_value=0.0, value=60000.0, step=1000.0)
            debt_ratio       = st.slider("Debt Ratio",              0.0, 1.0, 0.25, 0.01)
            monthly_expenses = st.number_input("Monthly Expenses ($)", min_value=0.0, value=2000.0)
        with col2:
            late_payments    = st.number_input("Late Payments",     min_value=0, max_value=20, value=0)
            num_loans        = st.number_input("Number of Loans",   min_value=0, max_value=15, value=1)
            dependents       = st.number_input("Dependents",        min_value=0, max_value=10, value=2)

        submitted = st.form_submit_button("💰 Estimate CLV", use_container_width=True)

    if submitted:
        input_dict = dict(age=age, income=income, debt_ratio=debt_ratio,
                          monthly_expenses=monthly_expenses, late_payments=late_payments,
                          num_loans=num_loans, dependents=dependents)
        try:
            clv = pr.predict_clv(input_dict)
            st.metric("Predicted Customer Lifetime Value", f"${clv:,.2f}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=clv,
                title={"text": "CLV ($)"},
                gauge={
                    "axis": {"range": [0, 400000]},
                    "bar": {"color": "#4f8ef7"},
                    "steps": [
                        {"range": [0, 100000],   "color": "#ffcccc"},
                        {"range": [100000, 250000], "color": "#fff3cc"},
                        {"range": [250000, 400000], "color": "#ccffcc"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        except FileNotFoundError:
            st.error("Models not found. Please run `python models.py` first.")
        except Exception as e:
            st.error(f"Error: {e}")


# Segmentation
elif page == "🗂️ Segmentation":
    st.title("🗂️ Customer Segmentation (K-Means)")
    st.write("Clusters all customers in the database into 4 risk/behaviour segments.")

    try:
        customers = db.get_all_customers()

        if len(customers) < 4:
            st.warning("Need at least 4 customers in the database to segment.")
        else:
            if st.button("🔄 Run Segmentation on All Customers"):
                labels = pr.predict_clusters_bulk(customers)
                for cid, label in zip(customers["id"], labels):
                    db.update_cluster(int(cid), int(label))
                st.success("✅ Clusters updated!")
                customers = db.get_all_customers()

            if not customers["cluster_label"].eq(-1).all():
                cluster_map = {0: "Low Risk", 1: "Medium Risk",
                               2: "High Risk", 3: "Very High Risk"}
                customers["Segment"] = customers["cluster_label"].map(cluster_map)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.scatter(customers, x="income", y="debt_ratio",
                                     color="Segment", hover_data=["age", "late_payments"],
                                     title="Income vs Debt Ratio by Segment",
                                     color_discrete_sequence=px.colors.qualitative.Set1)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig2 = px.scatter(customers, x="late_payments", y="num_loans",
                                      color="Segment", hover_data=["income"],
                                      title="Late Payments vs Loans by Segment",
                                      color_discrete_sequence=px.colors.qualitative.Set1)
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Segment Summary")
                summary = customers.groupby("Segment")[
                    ["income", "debt_ratio", "late_payments", "num_loans"]
                ].mean().round(2)
                st.dataframe(summary, use_container_width=True)

    except FileNotFoundError:
        st.error("Models not found. Please run `python models.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")


# Model Comparison
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")

    try:
        metrics = db.get_all_metrics()

        if metrics.empty:
            st.info("No metrics saved yet. Run `python models.py` and add customers + predictions first.")
        else:
            metrics = metrics[["model_name", "accuracy", "f1_score", "roc_auc"]]

            # Bar chart comparison
            fig = px.bar(
                metrics.melt(id_vars="model_name", var_name="Metric", value_name="Score"),
                x="model_name", y="Score", color="Metric", barmode="group",
                labels={"model_name": "Model"},
                title="Accuracy · F1 · ROC-AUC by Model",
                color_discrete_sequence=["#4f8ef7", "#f7a24f", "#4ff7a2"]
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(
                metrics.rename(columns={"model_name": "Model", "accuracy": "Accuracy",
                                         "f1_score": "F1 Score", "roc_auc": "ROC-AUC"})
                       .set_index("Model"),
                use_container_width=True
            )

        # Re-train button
        st.markdown("---")
        if st.button("🔄 Re-train All Models & Save Metrics"):
            with st.spinner("Training..."):
                from models import train_all
                m = train_all()
                for name, vals in m.items():
                    db.insert_metrics(name, vals["accuracy"], vals["f1_score"], vals["roc_auc"])
            st.success("✅ Models retrained and metrics saved!")
            st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")


# Customer Database
elif page == "🗄️ Customer Database":
    st.title("🗄️ Customer Database")

    try:
        customers = db.get_all_customers()

        if customers.empty:
            st.info("No customers yet. Add them via 'Predict Default'.")
        else:
            st.write(f"**{len(customers)} customers** in the database.")

            # Search
            search = st.text_input("Search by Customer ID")
            if search:
                customers = customers[customers["id"].astype(str).str.contains(search)]

            st.dataframe(customers, use_container_width=True)

            # View predictions for a customer
            st.markdown("---")
            st.subheader("View Predictions for a Customer")
            cid = st.number_input("Customer ID", min_value=1, step=1)
            if st.button("Fetch Predictions"):
                preds = db.get_predictions_for_customer(int(cid))
                if preds.empty:
                    st.warning("No predictions found for this customer.")
                else:
                    st.dataframe(preds, use_container_width=True)

            # Delete a customer
            st.markdown("---")
            st.subheader("Delete a Customer")
            del_id = st.number_input("Customer ID to Delete", min_value=1, step=1, key="del")
            if st.button("🗑️ Delete", type="primary"):
                db.delete_customer(int(del_id))
                st.success(f"Customer #{del_id} deleted.")
                st.rerun()

    except Exception as e:
        st.error(f"Database error: {e}")
        st.info("Make sure PostgreSQL is running and tables are created.")
