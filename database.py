import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from sqlalchemy import create_engine

# change these credentials
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "Finance Risk",
    "user":     "postgres",
    "password": "blah blah blah"   # 
}

def get_connection():
    """Raw psycopg2 connection — used for INSERT/UPDATE/DELETE."""
    return psycopg2.connect(**DB_CONFIG)

def get_engine():
    """SQLAlchemy engine — used for pd.read_sql()."""
    c = DB_CONFIG
    return create_engine(
        f"postgresql+psycopg2://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['dbname']}"
    )


# create tables
def create_tables():
    sql = """
    CREATE TABLE IF NOT EXISTS customers (
        id                  SERIAL PRIMARY KEY,
        age                 INT,
        income              FLOAT,
        debt_ratio          FLOAT,
        monthly_expenses    FLOAT,
        late_payments       INT,
        num_loans           INT,
        dependents          INT,
        cluster_label       INT DEFAULT -1,
        created_at          TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id                  SERIAL PRIMARY KEY,
        customer_id         INT REFERENCES customers(id),
        model_name          TEXT,
        prediction          TEXT,
        probability         FLOAT,
        predicted_at        TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS model_metrics (
        id                  SERIAL PRIMARY KEY,
        model_name          TEXT,
        accuracy            FLOAT,
        f1_score            FLOAT,
        roc_auc             FLOAT,
        trained_at          TIMESTAMP DEFAULT NOW()
    );
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Tables created successfully.")


# Customer Insertion into database
def insert_customer(age, income, debt_ratio, monthly_expenses,
                    late_payments, num_loans, dependents, cluster_label=-1):
    sql = """
    INSERT INTO customers
        (age, income, debt_ratio, monthly_expenses, late_payments, num_loans, dependents, cluster_label)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, (int(age), float(income), float(debt_ratio), float(monthly_expenses),
                      int(late_payments), int(num_loans), int(dependents), int(cluster_label)))
    customer_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return customer_id

def update_cluster(customer_id, cluster_label):
    sql = "UPDATE customers SET cluster_label = %s WHERE id = %s;"
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, (cluster_label, customer_id))
    conn.commit()
    cur.close()
    conn.close()


def delete_customer(customer_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE customer_id = %s;", (customer_id,))
    cur.execute("DELETE FROM customers WHERE id = %s;", (customer_id,))
    conn.commit()
    cur.close()
    conn.close()


# Model Predictions
def insert_prediction(customer_id, model_name, prediction, probability):
    sql = """
    INSERT INTO predictions (customer_id, model_name, prediction, probability)
    VALUES (%s, %s, %s, %s);
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, (customer_id, model_name, prediction, float(probability)))
    conn.commit()
    cur.close()
    conn.close()

# Model Metrics
def insert_metrics(model_name, accuracy, f1_score, roc_auc):
    sql = """
    INSERT INTO model_metrics (model_name, accuracy, f1_score, roc_auc)
    VALUES (%s, %s, %s, %s);
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, (model_name, float(accuracy), float(f1_score), float(roc_auc)))
    conn.commit()
    cur.close()
    conn.close()

# -------------------------------------------------------------

def get_all_customers():
    return pd.read_sql("SELECT * FROM customers ORDER BY created_at DESC", get_engine())

# -------------------------------------------------------------

def get_predictions_for_customer(customer_id):
    return pd.read_sql(
        "SELECT * FROM predictions WHERE customer_id = %s ORDER BY predicted_at DESC",
        get_engine(), params=(customer_id,)
    )

# -------------------------------------------------------------

def get_all_metrics():
    return pd.read_sql(
        "SELECT DISTINCT ON (model_name) * FROM model_metrics ORDER BY model_name, trained_at DESC",
        get_engine()
    )

# -------------------------------------------------------------

if __name__ == "__main__":
    create_tables()