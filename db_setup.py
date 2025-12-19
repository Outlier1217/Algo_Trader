import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dbname=os.getenv("DB_NAME"),
    port=os.getenv("DB_PORT")
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    initial_balance FLOAT,
    final_balance FLOAT,
    total_return FLOAT,
    max_drawdown FLOAT,
    actions_json TEXT,
    notes TEXT
);
""")

conn.commit()
cur.close()
conn.close()

print("✅ Database table ready!")
