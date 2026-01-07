# Day 1: Data Workflow - Project Setup and Basic I/O

## Learning Objectives

By the end of Day 1, you will be able to:
- Set up a professional data project structure
- Read CSV files with proper type handling
- Write Parquet files for processed outputs
- Create a basic ETL script that runs end-to-end

## Activities

### Task 1: Project Structure (10 minutes)

Create the standard folder layout for a data project:

```
your-project/
├── data/
│   ├── raw/          # Immutable input data
│   ├── cache/        # Cached API responses
│   ├── processed/    # Clean, analysis-ready outputs
│   └── external/      # Reference data
├── src/
│   └── data_workflow/  # Your package (or project_name/)
│       └── __init__.py
├── scripts/          # Run scripts
├── reports/
│   └── figures/      # Exported charts
├── pyproject.toml    # Dependencies
├── README.md         # Project documentation
└── .gitignore        # Git ignore file
```

### Task 2: Virtual Environment and Dependencies (15 minutes)

1. Create a virtual environment (at the project root):

```sh
uv init .
uv sync
```

2. Install dependencies:

```sh
uv add pandas pyarrow httpx
```

**References:**
- [uv documentation](https://docs.astral.sh/uv/)

### Task 3: Sample Raw Data (2 minutes)

Create two CSV files in `data/raw/`:

**`data/raw/orders.csv`:**
```csv
order_id,user_id,amount,quantity,created_at,status
A0001,0001,12.50,1,2025-12-01T10:05:00Z,Paid
A0002,0002,8.00,2,2025-12-01T11:10:00Z,paid
A0003,0003,not_a_number,1,2025-12-02T09:00:00Z,Refund
A0004,0001,25.00,,2025-12-03T14:30:00Z,PAID
A0005,0004,100.00,1,not_a_date,paid
```

**`data/raw/users.csv`:**
```csv
user_id,country,signup_date
0001,SA,2025-11-15
0002,SA,2025-11-20
0003,AE,2025-11-22
0004,SA,2025-11-25
```

### Task 4: Implement `config.py` (10 minutes)

Create `src/data_workflow/config.py` with:

1. A `Paths` dataclass containing:
   - `root: Path`
   - `raw: Path`
   - `cache: Path`
   - `processed: Path`
   - `external: Path`

2. A `make_paths(root: Path) -> Paths` function.

**References:**
- [dataclasses documentation](https://docs.python.org/3/library/dataclasses.html)
- [pathlib.Path operations](https://docs.python.org/3/library/pathlib.html#operators)

**Key Concepts:**
- Use `@dataclass(frozen=True)` to make Paths immutable
- Use `Path` objects (not strings) for cross-platform compatibility
- Use `/` operator to join paths: `root / "data" / "raw"`

### Task 5: Implement `io.py` (20 minutes)

Create `src/data_workflow/io.py` with these functions:

1. **`read_orders_csv(path: Path) -> pd.DataFrame`**
   - Read CSV with `dtype={"order_id": "string", "user_id": "string"}`
   - Handle missing values: `na_values=["", "NA", "N/A", "null", "None"]`
   - Keep default NA markers: `keep_default_na=True`

2. **`read_users_csv(path: Path) -> pd.DataFrame`**
   - Similar to `read_orders_csv`, but only `user_id` needs to be string

3. **`write_parquet(df: pd.DataFrame, path: Path) -> None`**
   - Create parent directory if needed: `path.parent.mkdir(parents=True, exist_ok=True)`
   - Write with `df.to_parquet(path, index=False)`

4. **`read_parquet(path: Path) -> pd.DataFrame`**
   - Simply: `return pd.read_parquet(path)`

**References:**
- [pandas.read_csv documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [pandas.DataFrame.to_parquet](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html)
- [pandas.read_parquet](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html)

**Key Concepts:**
- **Why strings for IDs?** Leading zeros are preserved. "0001" stays "0001", not 1
- **Why Parquet?** Preserves types exactly, faster, smaller files
- **Why na_values?** Centralizes missing value handling across your project

### Task 6: Implement `transforms.py` with `enforce_schema` (15 minutes)

Create `src/data_workflow/transforms.py` with:

**`enforce_schema(df: pd.DataFrame) -> pd.DataFrame`**
- Convert `order_id` and `user_id` to strings
- Convert `amount` to `Float64` using `pd.to_numeric(..., errors="coerce")`
- Convert `quantity` to `Int64` using `pd.to_numeric(..., errors="coerce")`

**References:**
- [pandas.to_numeric](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html)
- [pandas.DataFrame.assign](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html)
- [pandas nullable integer types](https://pandas.pydata.org/docs/user_guide/integer_na.html)

**Key Concepts:**
- `errors="coerce"` converts invalid numbers to NaN (doesn't crash)
- `Float64` and `Int64` (capital) are nullable types that can hold NaN
- Regular `float64` and `int64` cannot hold NaN

### Task 7: Create `run_day1_load.py` Script (20 minutes)

Create `scripts/run_day1_load.py` that:

1. Sets up paths: `ROOT = Path(__file__).resolve().parents[1]`
2. Adds `src/` to Python path for imports as the first item in the path: `sys.path.insert(0, str(ROOT / "src"))`
3. Loads raw CSVs using your `io.py` functions
4. Applies `enforce_schema` to orders
5. Writes Parquet outputs to `data/processed/`
6. Logs row counts and output paths

**References:**
- [Python logging module](https://docs.python.org/3/library/logging.html)
- [pathlib.Path.resolve](https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve)

## Progressive Hints

### If tests fail for project structure:

**Hint 1:** Make sure you're in the project root directory when running tests. The tests look for files relative to the current directory. To check, run `pwd` (Linux/Mac) or `echo %cd%` (Windows) in the terminal and make sure it points to the project root directory.

**Hint 2:** Check that all required directories exist. Use `ls -R` (Linux/Mac) or `tree /F` (Windows) to see your directory structure.

**Hint 3:** Make sure `pyproject.toml` is in the root directory, not in a subdirectory.

### If tests fail for config.py:

**Hint 1:** Make sure you import `dataclass` from `dataclasses` and `Path` from `pathlib`.

**Hint 2:** The `Paths` class should be a frozen dataclass:
```python
@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    # ... etc
```

**Hint 3:** The `make_paths` function should construct paths like:
```python
def make_paths(root: Path) -> Paths:
    data = root / "data"
    return Paths(
        root=root,
        raw=data / "raw",
        # ... etc
    )
```

### If tests fail for io.py:

**Hint 1:** Make sure you're using `dtype={"order_id": "string", "user_id": "string"}` in `pd.read_csv()`. The word "string" (not "str") is important!

**Hint 2:** Define a constant `NA = ["", "NA", "N/A", "null", "None"]` at the top of your file, then use `na_values=NA` in `pd.read_csv()`.

**Hint 3:** For `write_parquet`, create the directory first:
```python
path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(path, index=False)
```

### If tests fail for enforce_schema:

**Hint 1:** Use `pd.to_numeric()` with `errors="coerce"` to convert strings to numbers safely.

**Hint 2:** After converting, cast to nullable types:
```python
amount=pd.to_numeric(df["amount"], errors="coerce").astype("Float64")
```

**Hint 3:** Use `.assign()` to create a new DataFrame with updated columns:
```python
return df.assign(
    order_id=df["order_id"].astype("string"),
    amount=pd.to_numeric(df["amount"], errors="coerce").astype("Float64"),
    # ... etc
)
```

### If your script doesn't run:

**Hint 1:** Make sure you add `src/` to `sys.path` before importing:

```python

# ORDER IS IMPORTANT
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Your project imports should be AFTER.
# Here:
from data_workflow.config import make_paths
from data_workflow.io import read_orders_csv

# ...
```

**Hint 2:** Run the script from the project root directory, not from inside `scripts/`.

**Hint 3:** Make sure your virtual environment is activated and has all dependencies installed.

## Full Solution Reference

If you're completely stuck, here are the key code snippets (but try the hints first!):


### config.py
```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    cache: Path
    processed: Path
    external: Path

def make_paths(root: Path) -> Paths:
    data = root / "data"
    return Paths(
        root=root,
        raw=data / "raw",
        cache=data / "cache",
        processed=data / "processed",
        external=data / "external",
    )
```

### io.py (key parts)
```python
from pathlib import Path
import pandas as pd

NA = ["", "NA", "N/A", "null", "None"]

def read_orders_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"order_id": "string", "user_id": "string"},
        na_values=NA,
        keep_default_na=True,
    )

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
```

### transforms.py
```python
import pandas as pd

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        order_id=df["order_id"].astype("string"),
        user_id=df["user_id"].astype("string"),
        amount=pd.to_numeric(df["amount"], errors="coerce").astype("Float64"),
        quantity=pd.to_numeric(df["quantity"], errors="coerce").astype("Int64"),
    )
```

## Checklist

Before submitting, make sure:

- [ ] All directories are created
- [ ] `pyproject.toml` has correct dependencies
- [ ] `README.md` has required sections
- [ ] `.gitignore` excludes `.venv` and `__pycache__`
- [ ] `config.py` has `Paths` class and `make_paths` function
- [ ] `io.py` has all 4 functions with correct signatures
- [ ] `transforms.py` has `enforce_schema` function
- [ ] `scripts/run_day1_load.py` runs without errors
- [ ] Raw data files exist in `data/raw/`
- [ ] Script creates `data/processed/orders.parquet`

## Next Steps

Once all is done, you're ready for Day 2! Day 2 will add data quality checks and cleaning functions.

