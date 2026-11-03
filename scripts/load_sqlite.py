"""Load root CSV inputs into a local SQLite database for the SQL examples."""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("members_v3", "transactions_v2", "user_logs_v2", "train_v2")

def main() -> None:
    db_path = ROOT / "kkbox_analytics.sqlite"
    with sqlite3.connect(db_path) as conn:
        for table in TABLES:
            csv_path = ROOT / f"{table}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            for chunk in pd.read_csv(csv_path, chunksize=100_000):
                chunk.to_sql(table, conn, if_exists="append" if conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone() else "replace", index=False)
            print(f"Loaded {table}")
    print(f"SQLite database created: {db_path}")

if __name__ == "__main__":
    main()
