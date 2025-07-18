import json
import pandas as pd
import logging

from core.session_store import SessionStore
from models.telemetry_data import TelemetryData

logger = logging.getLogger(__name__)
# -------------- helpers ----------------
def _sample_df(df: pd.DataFrame, keep_rows: int = 100) -> str:
    if len(df) <= keep_rows:
        return df.to_json(orient="records")
    step = max(1, len(df) // keep_rows)
    return df.iloc[::step].to_json(orient="records")

def _describe_df(df: pd.DataFrame) -> str:
    # only numeric columns
    return json.dumps(df.describe().to_dict(), default=str)

def summarise_df(df: pd.DataFrame, max_rows: int = 10000) -> str:
    n = len(df)
    logger.info(f"Summarising {n} rows")
    if n <= 500:
        return df.to_json(orient="records")

    if n <= max_rows:
        # Sample from start, middle, and end
        head = df.head(300)
        mid = df.iloc[n//2 - 150:n//2 + 150]
        tail = df.tail(300)
        merged = pd.concat([head, mid, tail]).drop_duplicates()
        return merged.to_json(orient="records")

    # For massive data: return hybrid of describe + sample
    stats = _describe_df(df)
    sample = _sample_df(df, keep_rows=300)
    
    return json.dumps({
        "sample": json.loads(sample),
        "summary": json.loads(stats),
        "note": f"Sampled from {n} rows to fit context limits"
    }, indent=2)

def build_context(tdata: TelemetryData, msg_types: set[str], session_id: str, store: SessionStore) -> str:
    parts = []
    for m in msg_types:
        cached = store.get_cached_context(session_id, m)
        if cached:
            logger.info(f"Using cached summary for: {m}")
            parts.append(f"### {m} (cached)\n{cached}")
            continue

        df = tdata.get_df(m)
        if df is None or df.empty:
            continue
        summary = summarise_df(df)
        store.cache_context(session_id, m, summary)
        parts.append(f"### {m} ({len(df)} rows)\n{summary}")
    return "\n\n".join(parts) or "No relevant data found."

