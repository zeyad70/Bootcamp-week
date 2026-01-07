# Day 3: Datetimes, Outliers, and Joins

## Learning Objectives

By the end of Day 3, you will be able to:
- Parse datetime strings and extract time components
- Handle outliers using IQR and winsorization
- Perform safe joins with validation
- Build an analytics-ready table

## Activities

### Task 1: Add Datetime Transforms (20 minutes)

In `src/data_workflow/transforms.py`, add:

1. **`parse_datetime(df: pd.DataFrame, col: str, *, utc: bool = True) -> pd.DataFrame`**
   - Convert text column to datetime using `pd.to_datetime(..., errors="coerce", utc=utc)`
   - Use `.assign()` to update the column

2. **`add_time_parts(df: pd.DataFrame, ts_col: str) -> pd.DataFrame`**
   - Extract: `date`, `year`, `month`, `dow` (day of week), `hour`
   - Use `.dt` accessor (only works on datetime columns!)

**References:**
- [pandas.to_datetime](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)
- [pandas.Series.dt accessor](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.html)
- [pandas.Series.dt.to_period](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.to_period.html)

**Key Concepts:**
- `.dt` accessor only works AFTER parsing to datetime
- `errors="coerce"` converts invalid dates to NaT (doesn't crash)
- `utc=True` stores as timezone-aware datetime

### Task 2: Add Outlier Helpers (20 minutes)

In `src/data_workflow/transforms.py`, add:

1. **`iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]`**
   - Calculate Q1, Q3, IQR
   - Return (lower_bound, upper_bound) using IQR method

2. **`winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series`**
   - Cap values at percentiles using `.clip()`

3. **`add_outlier_flag(df: pd.DataFrame, col: str, *, k: float = 1.5) -> pd.DataFrame`** (optional)
   - Flag outliers without removing them

**References:**
- [pandas.Series.quantile](https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html)
- [pandas.Series.clip](https://pandas.pydata.org/docs/reference/api/pandas.Series.clip.html)

### Task 3: Create `joins.py` (15 minutes)

Create `src/data_workflow/joins.py` with:

**`safe_left_join(left, right, on, validate, suffixes=...)`**
- Wrapper around `pd.merge()` with `validate` parameter
- Prevents join explosions by checking cardinality

**References:**
- [pandas.DataFrame.merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)
- [merge validate parameter](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge)

**Key Concepts:**
- `validate="many_to_one"`: left can have duplicates, right must be unique
- Always validate joins to prevent silent bugs

### Task 4: Build Analytics Table Script (30 minutes)

Create `scripts/run_day3_build_analytics.py` that:
1. Loads `orders_clean.parquet` and `users.parquet`
2. Validates inputs (columns, uniqueness)
3. Parses `created_at` and adds time parts
4. Joins orders → users (validate="many_to_one")
5. Winsorizes amount and adds outlier flag
6. Writes `analytics_table.parquet`

## Progressive Hints

### If datetime parsing fails:

**Hint 1:** Make sure to parse BEFORE using `.dt` accessor:
```python
df = parse_datetime(df, "created_at")  # Parse first
df = add_time_parts(df, "created_at")  # Then extract parts
```

**Hint 2:** Use `errors="coerce"` to handle invalid dates gracefully.

### If joins fail:

**Hint 1:** Always validate uniqueness on the "right" table before joining:
```python
assert_unique_key(users, "user_id")  # Before join
```

**Hint 2:** Check row counts after join:
```python
assert len(joined) == len(orders), "Row count changed (join explosion?)"
```

## Checklist

- [ ] `transforms.py` has datetime and outlier functions
- [ ] `joins.py` has `safe_left_join` function
- [ ] `scripts/run_day3_build_analytics.py` runs successfully
- [ ] Creates `data/processed/analytics_table.parquet`

