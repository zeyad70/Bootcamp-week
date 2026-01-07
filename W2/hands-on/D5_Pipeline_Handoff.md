# Day 5: ETL Pipeline and Handoff

## Learning Objectives

By the end of Day 5, you will be able to:
- Package your work as a production-ready ETL pipeline
- Generate run metadata for reproducibility
- Write a clear summary with findings and caveats
- Ensure your repo is ready for handoff

## Activities

### Task 1: Create ETL Module (60 minutes)

Create `src/data_workflow/etl.py` with:

1. **`ETLConfig` dataclass** - Configuration with all paths
2. **`load_inputs(cfg)`** - Extract raw data
3. **`transform(orders, users)`** - Clean and enrich data
4. **`load_outputs(analytics, users, cfg)`** - Write processed outputs
5. **`write_run_meta(cfg, analytics)`** - Write run metadata JSON
6. **`run_etl(cfg)`** - Orchestrate the pipeline

**References:**
- [Python logging module](https://docs.python.org/3/library/logging.html)
- [dataclasses.asdict](https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict)
- [JSON module](https://docs.python.org/3/library/json.html)

### Task 2: Create Main ETL Script (15 minutes)

Create `scripts/run_etl.py` that:
- Defines `ROOT` path
- Creates `ETLConfig` with all paths
- Calls `run_etl(cfg)`

### Task 3: Write Summary (25 minutes)

Create `reports/summary.md` with:
- **Key Findings**: Bulleted, quantified results
- **Definitions**: Metrics and filters used
- **Data Quality Caveats**: Missingness, duplicates, join coverage, outliers
- **Next Questions**: Recommended follow-up analyses

## Progressive Hints

### If ETL fails:

**Hint 1:** Make sure all Day 1-3 functions are working first. Test them individually before combining.

**Hint 2:** Add logging to see where it fails:
```python
log.info("Starting transform...")
analytics = transform(orders, users)
log.info("Transform complete: %s rows", len(analytics))
```

**Hint 3:** Check that all imports are correct. Make sure you're importing from the right module (`data_workflow` or `project_name`).

**Hint 4:** Verify that your transform function calls all the right functions in the right order. Compare with your Day 1-3 scripts.

**Hint 5:** If you get "module not found" errors, check that `src/` is in `sys.path` and your package structure is correct.

### If metadata writing fails:

**Hint 1:** Create parent directory first:
```python
cfg.run_meta.parent.mkdir(parents=True, exist_ok=True)
```

**Hint 2:** Convert Path objects to strings for JSON:
```python
"config": {k: str(v) for k, v in asdict(cfg).items()}
```

**Hint 3:** Make sure you're using `json.dumps()` with `indent=2` for readable output.

**Hint 4:** Check that all values in the metadata dictionary are JSON-serializable (no Path objects, no NaN values - convert to int/float/str).

### If script doesn't run:

**Hint 1:** Make sure you're running from the project root directory, not from inside `scripts/`.

**Hint 2:** Check that `ROOT` path is correct. Use `print(ROOT)` to verify.

**Hint 3:** Verify that `src/` is added to `sys.path` before importing:
```python
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

**Hint 4:** Make sure your package name matches (might be `data_workflow` or `project_name`).

**Hint 5:** Check that all required files exist (raw data files, etc.).

### If summary.md is incomplete:

**Hint 1:** Look at your EDA notebook - all findings should come from there.

**Hint 2:** Quantify everything - use actual numbers from your analysis.

**Hint 3:** Be specific about data quality issues - check your missingness reports and join coverage.

**Hint 4:** Think about what someone else would need to know to understand and reproduce your work.

## Full Solution Reference

If you're completely stuck, here are the key code snippets (but try the hints first!):

### etl.py (key parts)
```python
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from data_workflow.io import read_orders_csv, read_users_csv, write_parquet
from data_workflow.joins import safe_left_join
from data_workflow.quality import assert_non_empty, assert_unique_key, require_columns
from data_workflow.transforms import (
    add_missing_flags, add_outlier_flag, add_time_parts,
    apply_mapping, enforce_schema, normalize_text,
    parse_datetime, winsorize,
)

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class ETLConfig:
    """Configuration for ETL pipeline run."""
    root: Path
    raw_orders: Path
    raw_users: Path
    out_orders_clean: Path
    out_users: Path
    out_analytics: Path
    run_meta: Path

def load_inputs(cfg: ETLConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract: Load raw input files."""
    orders = read_orders_csv(cfg.raw_orders)
    users = read_users_csv(cfg.raw_users)
    return orders, users

def transform(orders_raw: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Transform: Clean, validate, and enrich data."""
    # Validate inputs
    require_columns(orders_raw, ["order_id", "user_id", "amount", "quantity", "created_at", "status"])
    require_columns(users, ["user_id", "country", "signup_date"])
    assert_non_empty(orders_raw, "orders_raw")
    assert_non_empty(users, "users")
    assert_unique_key(users, "user_id")
    
    # Enforce schema
    orders = enforce_schema(orders_raw)
    
    # Clean text
    status_norm = normalize_text(orders["status"])
    mapping = {"paid": "paid", "refund": "refund", "refunded": "refund"}
    orders = orders.assign(status_clean=apply_mapping(status_norm, mapping))
    
    # Add missing flags
    orders = add_missing_flags(orders, cols=["amount", "quantity"])
    
    # Parse datetime and extract time parts
    orders = parse_datetime(orders, col="created_at", utc=True)
    orders = add_time_parts(orders, ts_col="created_at")
    
    # Join with users
    joined = safe_left_join(orders, users, on="user_id", validate="many_to_one", suffixes=("", "_user"))
    assert len(joined) == len(orders), "Row count changed (join explosion?)"
    
    # Handle outliers
    joined = joined.assign(amount_winsor=winsorize(joined["amount"]))
    joined = add_outlier_flag(joined, "amount", k=1.5)
    
    return joined

def load_outputs(analytics: pd.DataFrame, users: pd.DataFrame, cfg: ETLConfig) -> None:
    """Load: Write processed outputs to disk."""
    write_parquet(users, cfg.out_users)
    write_parquet(analytics, cfg.out_analytics)

def write_run_meta(cfg: ETLConfig, *, analytics: pd.DataFrame) -> None:
    """Write run metadata for reproducibility."""
    missing_created_at = int(analytics["created_at"].isna().sum())
    country_match_rate = 1.0 - float(analytics["country"].isna().mean())
    
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rows_out": int(len(analytics)),
        "missing_created_at": missing_created_at,
        "country_match_rate": country_match_rate,
        "config": {k: str(v) for k, v in asdict(cfg).items()},
    }
    
    cfg.run_meta.parent.mkdir(parents=True, exist_ok=True)
    cfg.run_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def run_etl(cfg: ETLConfig) -> None:
    """Run the complete ETL pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    
    log.info("Loading inputs")
    orders_raw, users = load_inputs(cfg)
    
    log.info("Transforming (orders=%s, users=%s)", len(orders_raw), len(users))
    analytics = transform(orders_raw, users)
    
    log.info("Writing outputs to %s", cfg.out_analytics.parent)
    load_outputs(analytics, users, cfg)
    
    log.info("Writing run metadata: %s", cfg.run_meta)
    write_run_meta(cfg, analytics=analytics)
    
    log.info("ETL complete: %s rows in analytics table", len(analytics))
```

### run_etl.py
```python
"""Main ETL entrypoint script."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_workflow.etl import ETLConfig, run_etl

cfg = ETLConfig(
    root=ROOT,
    raw_orders=ROOT / "data" / "raw" / "orders.csv",
    raw_users=ROOT / "data" / "raw" / "users.csv",
    out_orders_clean=ROOT / "data" / "processed" / "orders_clean.parquet",
    out_users=ROOT / "data" / "processed" / "users.parquet",
    out_analytics=ROOT / "data" / "processed" / "analytics_table.parquet",
    run_meta=ROOT / "data" / "processed" / "_run_meta.json",
)

if __name__ == "__main__":
    run_etl(cfg)
```

### summary.md (template)
```markdown
# Summary of Findings and Caveats

## Key Findings

- **Finding 1 (quantified)**: [Example: SA accounts for 60% of total revenue with $X,XXX]
- **Finding 2 (quantified)**: [Example: Monthly revenue increased by X% from Month A to Month B]
- **Finding 3 (quantified)**: [Example: Average order value is $XX.XX, with median $XX.XX]
- **Finding 4 (quantified)**: [Example: Refund rate differs by X percentage points between countries]

## Definitions

- **Revenue**: Sum of `amount` column over all orders (excluding refunds if filtered)
- **AOV (Average Order Value)**: Mean of `amount` column
- **Refund rate**: Proportion of orders where `status_clean == "refund"`
- **Time window**: [Specify the date range of your data, e.g., "December 2025"]
- **Winsorized amount**: Order amounts capped at 1st and 99th percentiles to reduce outlier impact on visualizations

## Data Quality Caveats

### Missingness
- [Example: X% of orders have missing `created_at` values due to invalid date formats in raw data]
- [Example: Y% of orders have missing `quantity` values]

### Duplicates
- [Note if any duplicates were found and how they were handled]

### Join Coverage
- [Example: Z% of orders successfully matched to users table; unmatched orders may bias country-level results]

### Outliers
- [Example: X orders have amounts > $XXX, flagged as outliers; charts use winsorized values for readability]

### Other Issues
- [Any other data quality concerns, e.g., inconsistent status values, timezone issues, etc.]

## Next Questions

- [Example: How does refund rate vary by month?]
- [Example: What is the customer lifetime value by country?]
- [Example: Are there seasonal patterns in order volume?]
- [Example: What factors predict high-value orders?]

## Technical Notes

- **ETL Pipeline**: Run `uv run python scripts/run_etl.py` to reproduce processed outputs
- **Run Metadata**: See `data/processed/_run_meta.json` for run details
- **Data Source**: Raw data in `data/raw/`, processed outputs in `data/processed/`
- **EDA Notebook**: See `notebooks/eda.ipynb` for detailed analysis
```

## Checklist

- [ ] `etl.py` has complete ETL pipeline
- [ ] `scripts/run_etl.py` runs successfully
- [ ] Creates `data/processed/_run_meta.json`
- [ ] `reports/summary.md` has all required sections

