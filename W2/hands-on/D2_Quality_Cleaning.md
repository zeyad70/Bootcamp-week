# Day 2: Data Quality and Cleaning

## Learning Objectives

By the end of Day 2, you will be able to:
- Write data quality checks that fail fast
- Create missingness reports
- Normalize text data
- Add missing value flags
- Create a cleaning script that produces clean outputs

## Activities

### Task 1: Create `quality.py` (20 minutes)

Create `src/data_workflow/quality.py` with 4 validation functions:

1. **`require_columns(df: pd.DataFrame, cols: list[str]) -> None`**
   - Check that all required columns exist
   - Raise AssertionError with clear message if any are missing

2. **`assert_non_empty(df: pd.DataFrame, name: str = "df") -> None`**
   - Check that DataFrame has at least one row
   - Raise AssertionError if empty

3. **`assert_unique_key(df: pd.DataFrame, key: str, *, allow_na: bool = False) -> None`**
   - Check that a column contains unique values
   - Check for missing values if `allow_na=False`
   - Raise AssertionError if duplicates or missing values found

4. **`assert_in_range(s: pd.Series, lo=None, hi=None, name: str = "value") -> None`**
   - Check that all values are within a range
   - Ignore missing values (only check non-missing)
   - Raise AssertionError if values are outside range

**References:**
- [pandas.DataFrame.columns](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html)
- [pandas.Series.duplicated](https://pandas.pydata.org/docs/reference/api/pandas.Series.duplicated.html)
- [pandas.Series.notna](https://pandas.pydata.org/docs/reference/api/pandas.Series.notna.html)

**Key Concepts:**
- **Fail fast**: Check assumptions early, before doing expensive operations
- **Clear error messages**: Tell the user exactly what's wrong
- **Ignore missing**: For range checks, only validate non-missing values

### Task 2: Add Missingness Helpers (15 minutes)

In `src/data_workflow/transforms.py`, add:

1. **`missingness_report(df: pd.DataFrame) -> pd.DataFrame`**
   - Count missing values per column
   - Calculate percentage missing
   - Sort by percentage (most missing first)
   - Return DataFrame with columns: `n_missing`, `p_missing`

2. **`add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame`**
   - Add boolean columns indicating missing values
   - Column names: `{col}__isna` (e.g., `amount__isna`)
   - Don't drop rows - just flag them

**References:**
- [pandas.DataFrame.isna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html)
- [pandas.Series.sum](https://pandas.pydata.org/docs/reference/api/pandas.Series.sum.html)
- [pandas.DataFrame.assign](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html)

### Task 3: Add Text Normalization (20 minutes)

In `src/data_workflow/transforms.py`, add:

1. **`normalize_text(s: pd.Series) -> pd.Series`**
   - Trim whitespace: `.str.strip()`
   - Convert to lowercase: `.str.casefold()`
   - Collapse multiple spaces: `.str.replace()` with regex

2. **`apply_mapping(s: pd.Series, mapping: dict[str, str]) -> pd.Series`**
   - Map values using a dictionary
   - Values not in mapping stay unchanged

**References:**
- [pandas.Series.str.strip](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.strip.html)
- [pandas.Series.str.casefold](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.casefold.html)
- [pandas.Series.str.replace](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.replace.html)
- [pandas.Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html)
- [Python re module](https://docs.python.org/3/library/re.html)

### Task 4: Add Deduplication Helper (10 minutes)

In `src/data_workflow/transforms.py`, add:

**`dedupe_keep_latest(df: pd.DataFrame, key_cols: list[str], ts_col: str) -> pd.DataFrame`**
- Sort by timestamp
- Remove duplicates, keeping the latest row
- Reset index

**References:**
- [pandas.DataFrame.sort_values](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html)
- [pandas.DataFrame.drop_duplicates](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)

### Task 5: Create `run_day2_clean.py` (25 minutes)

Create a script that:
1. Loads raw CSVs
2. Runs quality checks (columns, non-empty)
3. Enforces schema
4. Creates missingness report → saves to `reports/missingness_orders.csv`
5. Normalizes status field → creates `status_clean`
6. Adds missing flags for `amount` and `quantity`
7. Validates ranges (amount >= 0, quantity >= 0)
8. Writes `orders_clean.parquet`

## Progressive Hints

### If quality checks fail:

**Hint 1:** Use simple [`assert` statements](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement) with clear error messages:
```python
assert len(df) > 0, f"{name} has 0 rows"
```

**Hint 2:** For `require_columns`, find missing columns first:
```python
missing = [c for c in cols if c not in df.columns]
assert not missing, f"Missing columns: {missing}"
```

**Hint 3:** For `assert_unique_key`, check duplicates:
```python
dup = df[key].duplicated(keep=False) & df[key].notna()
assert not dup.any(), f"{key} not unique; {dup.sum()} duplicate rows"
```

### If missingness_report fails:

**Hint 1:** Start with counting missing values:
```python
missing_counts = df.isna().sum()
```

**Hint 2:** Convert to DataFrame and add percentage:
```python
missing_counts.rename("n_missing").to_frame().assign(
    p_missing=lambda t: t["n_missing"] / len(df)
)
```

### If normalize_text fails:

**Hint 1:** Use pandas string methods in sequence:
```python
s.astype("string").str.strip().str.casefold()
```

**Hint 2:** For collapsing spaces, use regex:
```python
import re
_ws = re.compile(r"\s+")
s.str.replace(_ws, " ", regex=True)
```

## Full Solution Reference

### quality.py (key parts)
```python
def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def assert_non_empty(df: pd.DataFrame, name: str = "df") -> None:
    assert len(df) > 0, f"{name} has 0 rows"

def assert_unique_key(df: pd.DataFrame, key: str, *, allow_na: bool = False) -> None:
    if not allow_na:
        assert df[key].notna().all(), f"{key} contains NA"
    dup = df[key].duplicated(keep=False) & df[key].notna()
    assert not dup.any(), f"{key} not unique; {dup.sum()} duplicate rows"
```

### transforms.py additions
```python
def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna().sum()
        .rename("n_missing")
        .to_frame()
        .assign(p_missing=lambda t: t["n_missing"] / len(df))
        .sort_values("p_missing", ascending=False)
    )

def normalize_text(s: pd.Series) -> pd.Series:
    import re
    _ws = re.compile(r"\s+")
    return (
        s.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(_ws, " ", regex=True)
    )
```

## Checklist

- [ ] `quality.py` has all 4 functions
- [ ] `transforms.py` has all Day 2 functions
- [ ] `scripts/run_day2_clean.py` runs without errors
- [ ] Creates `data/processed/orders_clean.parquet`
- [ ] Creates `reports/missingness_orders.csv`

