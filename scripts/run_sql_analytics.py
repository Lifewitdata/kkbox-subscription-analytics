"""Load the canonical synthetic CSVs into SQLite and persist SQL query outputs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("members_v3", "transactions_v2", "user_logs_v2", "train_v2")
QUERY_FILES = (
    "01_cohort_retention.sql",
    "02_churn_by_segment.sql",
    "03_ltv_by_channel.sql",
    "04_data_quality_checks.sql",
)


def load_database(db_path: Path | None = None) -> Path:
    db_path = db_path or (ROOT / "kkbox_analytics.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        for table in TABLES:
            source = ROOT / f"{table}.csv"
            if not source.is_file():
                raise FileNotFoundError(f"Required source file missing: {source}")
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            first = True
            for chunk in pd.read_csv(source, chunksize=50_000):
                chunk.to_sql(table, connection, if_exists="replace" if first else "append",
                             index=False)
                first = False
    return db_path


def split_statements(sql: str) -> list[str]:
    # The committed SQL uses semicolon-terminated statements and no semicolons
    # inside literals; keeping the splitter simple makes query execution explicit.
    return [part.strip() for part in sql.split(";") if part.strip()]


def run_queries(db_path: Path | None = None, output_dir: Path | None = None) -> dict:
    db_path = db_path or load_database()
    output_dir = output_dir or (ROOT / "results" / "sql")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    with sqlite3.connect(db_path) as connection:
        for filename in QUERY_FILES:
            source = ROOT / "sql" / filename
            statements = split_statements(source.read_text(encoding="utf-8"))
            frames = []
            for index, statement in enumerate(statements, start=1):
                frame = pd.read_sql_query(statement, connection)
                frame.insert(0, "query_number", index)
                frames.append(frame)
            combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
            output_name = filename.replace(".sql", ".csv")
            combined.to_csv(output_dir / output_name, index=False)
            summary[filename] = {
                "statement_count": len(statements),
                "result_rows": int(len(combined)),
                "output_file": output_name,
            }
    (output_dir / "execution_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    database = load_database()
    summary = run_queries(database)
    print(json.dumps({"database": str(database), "queries": summary}, indent=2))


if __name__ == "__main__":
    main()
