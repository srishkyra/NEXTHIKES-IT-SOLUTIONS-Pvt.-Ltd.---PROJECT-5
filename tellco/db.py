"""Database export of the final scored table (Task 4.6)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import PIPELINE_VERSION

EXPORT_COLS = ["Engagement Score", "Experience Score", "Satisfaction Score",
               "Engagement Tier", "Experience Group", "Satisfaction Group"]


def scores_frame(comb: pd.DataFrame) -> pd.DataFrame:
    """The table to export: customer id, the three scores, segment labels, scoring version and timestamp."""
    f = comb[EXPORT_COLS].reset_index()
    f["MSISDN/Number"] = f["MSISDN/Number"].astype("int64")
    f["Scoring Version"] = str(PIPELINE_VERSION)
    f["Scored At"] = datetime.now().isoformat(timespec="seconds")
    return f


def validate_scores(frame: pd.DataFrame) -> None:
    """Raise ValueError unless: one row per customer, no missing values, scores finite and in range."""
    problems = []
    if frame["MSISDN/Number"].duplicated().any():
        problems.append("duplicate customer ids")
    if frame.isna().any().any():
        problems.append("missing values")
    scores = frame[["Engagement Score", "Experience Score", "Satisfaction Score"]].to_numpy(float)
    if not np.isfinite(scores).all():
        problems.append("non-finite scores")
    elif (scores[:, :2] < 0).any():
        problems.append("negative engagement or experience score")
    if not frame["Satisfaction Score"].between(0, 1).all():
        problems.append("satisfaction score outside 0-1")
    if problems:
        raise ValueError("Scored table failed validation: " + "; ".join(problems))


def mysql_url(user: str, password: str, host: str = "localhost", port: int | str = 3306, db: str = "tellco") -> str:
    from urllib.parse import quote_plus
    return f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


def export_scores(frame: pd.DataFrame, url: str, table: str = "user_scores", log_path: str | Path | None = None,
                  preview_rows: int = 10) -> dict:
    """Validate and write ``frame`` to ``table`` (replace); return a verification block with a real SELECT result."""
    from sqlalchemy import create_engine, text
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("Table name must be letters, digits and underscores only")
    validate_scores(frame)
    engine = create_engine(url)
    try:
        # 2,000 rows x 9 columns stays under SQLite's 32,766-variable limit (MySQL has no such limit here).
        frame.to_sql(table, engine, if_exists="replace", index=False, chunksize=2000, method="multi")
        prep = engine.dialect.identifier_preparer
        qt, qs = prep.quote(table), prep.quote("Satisfaction Score")
        query = f"SELECT * FROM {qt} ORDER BY {qs} DESC LIMIT {int(preview_rows)}"
        with engine.connect() as conn:
            preview = pd.read_sql(text(query), conn)
            n = int(conn.execute(text(f"SELECT COUNT(*) FROM {qt}")).scalar())
            server = conn.execute(text("SELECT VERSION()")).scalar() if engine.dialect.name == "mysql" else "sqlite"
    finally:
        engine.dispose()
    info = {"exported_at": datetime.now().isoformat(timespec="seconds"), "dialect": engine.dialect.name,
            "server_version": str(server), "table": table, "rows_written": len(frame), "rows_in_table": n,
            "select_query": query, "select_output": json.loads(preview.to_json(orient="records"))}
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(info, indent=2, default=str))
    return info