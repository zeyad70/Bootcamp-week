# Day 4: EDA and Visualization with Plotly

## Learning Objectives

By the end of Day 4, you will be able to:
- Create clear, publication-ready charts with Plotly
- Export figures to files
- Compute bootstrap confidence intervals
- Answer business questions with data

## Activities

### Task 1: Create Visualization Helpers (20 minutes)

Create `src/data_workflow/viz.py` with:

1. **`bar_sorted(df, x, y, title)`** - Sorted bar chart
2. **`time_line(df, x, y, title)`** - Line chart for trends
3. **`histogram_chart(df, x, nbins, title)`** - Distribution histogram
4. **`save_fig(fig, path, scale=2)`** - Export figure to PNG

**References:**
- [plotly.express documentation](https://plotly.com/python/plotly-express/)
- [plotly figure.update_layout](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_layout)
- [plotly figure.write_image](https://plotly.com/python/static-image-export/)

**Key Concepts:**
- Use `plotly.express` for quick charts
- Customize with `fig.update_layout()` and `fig.update_xaxes/yaxes()`
- Export requires `kaleido` package: `pip install kaleido`

### Task 2: Create Bootstrap Function (15 minutes)

Create `src/data_workflow/utils.py` with:

**`bootstrap_diff_means(a, b, n_boot=2000, seed=0)`**
- Resample each group with replacement
- Compute difference in means for each resample
- Return observed difference and 95% confidence interval

**References:**
- [numpy.random.default_rng](https://numpy.org/doc/stable/reference/random/generator.html)
- [numpy.quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
- [numpy.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html)

### Task 3: Create EDA Notebook (60 minutes)

Create `notebooks/eda.ipynb` that:
1. Loads `analytics_table.parquet`
2. Performs data audit (rows, dtypes, missingness)
3. Answers 3-6 questions with:
   - Summary tables
   - Visualizations
   - Interpretations and caveats
4. Includes one bootstrap comparison
5. Exports at least 3 figures to `reports/figures/`

## Progressive Hints

### If charts don't export:

**Hint 1:** Install kaleido: `uv add kaleido`

**Hint 2:** Make sure `reports/figures/` directory exists before exporting. Use `path.parent.mkdir(parents=True, exist_ok=True)` in `save_fig()`.

**Hint 3:** Check that you're using `fig.write_image(str(path), scale=scale)` - the path must be a string, not a Path object.

### If visualization functions fail:

**Hint 1:** Make sure you're importing plotly.express: `import plotly.express as px`

**Hint 2:** For `bar_sorted`, sort the DataFrame first:
```python
d = df.sort_values(y, ascending=False)
fig = px.bar(d, x=x, y=y, title=title)
```

**Hint 3:** For `time_line`, make sure your DataFrame is sorted by the time column before calling the function.

**Hint 4:** Check that column names match exactly (case-sensitive). Use `df.columns` to see available columns.

### If bootstrap fails:

**Hint 1:** Use fixed seed for reproducibility: `seed=0` in the function call.

**Hint 2:** Drop missing values before resampling:
```python
a_clean = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
b_clean = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
```

**Hint 3:** Check that both groups have data after cleaning:
```python
assert len(a_clean) > 0 and len(b_clean) > 0, "Empty group after cleaning"
```

**Hint 4:** Use `np.random.default_rng(seed)` for the random number generator (not `np.random.seed()`).

**Hint 5:** Make sure you're using `replace=True` in `rng.choice()` - this is bootstrap resampling with replacement.

### If notebook imports fail:

**Hint 1:** Make sure you add `src/` to `sys.path` before importing:
```python
ROOT = Path().resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
```

**Hint 2:** Check that your package name matches (might be `data_workflow` or `project_name` depending on your setup).

**Hint 3:** Run the notebook from the `notebooks/` directory, or adjust the `ROOT` path accordingly.

### If figures don't display in notebook:

**Hint 1:** Make sure you call the figure at the end of the cell (Jupyter displays the last expression).

**Hint 2:** Use `fig.show()` if the figure doesn't display automatically.

## Full Solution Reference

If you're completely stuck, here are the key code snippets (but try the hints first!):

### viz.py (key parts)
```python
from pathlib import Path
import plotly.express as px
import pandas as pd

def save_fig(fig, path: Path, *, scale: int = 2) -> None:
    """Save a Plotly figure as an image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=scale)

def bar_sorted(df: pd.DataFrame, x: str, y: str, title: str):
    """Create a sorted bar chart."""
    d = df.sort_values(y, ascending=False)
    fig = px.bar(d, x=x, y=y, title=title)
    fig.update_layout(
        title={"x": 0.02},
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
    )
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return fig

def time_line(df: pd.DataFrame, x: str, y: str, color=None, title: str = ""):
    """Create a line chart for time series data."""
    fig = px.line(df, x=x, y=y, color=color, title=title)
    fig.update_layout(title={"x": 0.02})
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return fig

def histogram_chart(df, x: str, nbins: int = 30, title: str = ""):
    """Create a histogram for distribution visualization."""
    fig = px.histogram(df, x=x, nbins=nbins, title=title)
    fig.update_layout(title={"x": 0.02})
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text="Number of orders")
    return fig
```

### utils.py (key parts)
```python
import numpy as np
import pandas as pd

def bootstrap_diff_means(a: pd.Series, b: pd.Series, *, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Compute bootstrap confidence interval for difference in means."""
    a_clean = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b_clean = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    
    assert len(a_clean) > 0 and len(b_clean) > 0, "Empty group after cleaning"
    
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a_clean, size=len(a_clean), replace=True)
        sb = rng.choice(b_clean, size=len(b_clean), replace=True)
        diffs.append(sa.mean() - sb.mean())
    
    diffs = np.array(diffs)
    return {
        "diff_mean": float(a_clean.mean() - b_clean.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }
```

### Notebook example (key parts)
```python
# Setup
from pathlib import Path
import pandas as pd
import sys

ROOT = Path().resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data_workflow.viz import bar_sorted, time_line, histogram_chart, save_fig
from data_workflow.utils import bootstrap_diff_means

DATA = ROOT / "data/processed/analytics_table.parquet"
FIGS = ROOT / "reports/figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet(DATA)

# Question 1: Revenue by country
rev = df.groupby("country", dropna=False)["amount"].sum().reset_index()
fig = bar_sorted(rev, "country", "amount", "Revenue by country (all time)")
save_fig(fig, FIGS / "revenue_by_country.png")
fig

# Question 2: Monthly trend
trend = df.groupby("month", dropna=False)["amount"].sum().reset_index().sort_values("month")
fig = time_line(trend, "month", "amount", title="Revenue over time (monthly)")
save_fig(fig, FIGS / "revenue_trend_monthly.png")
fig

# Question 3: Distribution
fig = histogram_chart(df, "amount_winsor", nbins=30, title="Order amount distribution")
save_fig(fig, FIGS / "amount_hist_winsor.png")
fig

# Question 4: Bootstrap comparison
d = df.assign(is_refund=(df["status_clean"] == "refund").astype(int))
a = d.loc[d["country"] == "SA", "is_refund"]
b = d.loc[d["country"] == "AE", "is_refund"]
result = bootstrap_diff_means(a, b)
print(f"Difference: {result['diff_mean']:.4f}")
print(f"95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
```

## Checklist

- [ ] `viz.py` has visualization functions
- [ ] `utils.py` has bootstrap function
- [ ] `notebooks/eda.ipynb` exists and runs
- [ ] At least 3 figures exported to `reports/figures/`

## Next Steps

You're ready for Day 5! Day 5 will package everything into a production-ready ETL pipeline and create a summary document for handoff.

