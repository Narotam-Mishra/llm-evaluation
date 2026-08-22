# How `evaluator.py` Works

## Purpose

`evaluator.py` decides whether the result of a model-generated SQL query is
equivalent to the expected (gold) SQL result.

It compares two pandas DataFrames:

- `gold_df`: the expected query result
- `generated_df`: the model-generated query result

The evaluator compares returned **values**, not SQL text or column names. It can
also accept a result with exactly one extra or one missing column. This is useful
when one correct query returns only a player's name while another returns the
player's name and score.

This module does **not**:

- ask an LLM to generate SQL;
- execute SQL against SQLite;
- read `golden_hard.csv`; or
- calculate an aggregate accuracy score.

Those orchestration steps must happen elsewhere. This module only compares two
already-created DataFrames and reports whether they match.

## High-level execution flow

```text
evaluate_one(gold_df, generated_df)
    |
    +-- Is generated_df None? -- yes --> incorrect: "sql_error"
    |
    +-- no --> compare_dataframes(...)
                 |
                 +-- Different row counts --> False
                 |
                 +-- Same column count --> canonicalize rows
                 |                          and compare strictly
                 |
                 +-- Column counts differ by 1 --> check whether every
                 |                                smaller row is a subset
                 |                                of its larger row
                 |
                 +-- Column counts differ by > 1 --> False
```

## Step-by-step explanation

### 1. Import pandas

```python
import pandas as pd
```

Pandas supplies the DataFrame structure and `pd.isna()`, which is used when
normalizing missing floating-point values such as `NaN`.

### 2. Normalize one value with `_canon_cell`

```python
def _canon_cell(v):
```

Database drivers and pandas can represent logically equal values with different
Python types. For example, SQLite might return `30`, `30.0`, or even `"30"`.
This function converts each cell into a canonical string before comparison.

Its rules are:

| Input | Canonical output | Reason |
|---|---|---|
| `None` or floating `NaN` | `"∅"` | Treat missing values consistently |
| `True` | `"str:True"` | Prevent Python's `True == 1` behavior |
| `30` | `"num:30"` | Normalize integers |
| `30.0` | `"num:30"` | Make integer-like floats equal to integers |
| `2.345678` | `"num:2.3457"` | Tolerate small floating-point differences |
| `" 30 "` | `"num:30"` | Trim and recognize numeric strings |
| `"Chahal"` | `"str:Chahal"` | Preserve ordinary text as text |

Floating-point values are rounded to four decimal places. Therefore, values
that differ only beyond the fourth decimal place are considered equal.

### 3. Convert rows to sets with `_rows_as_sets`

```python
def _rows_as_sets(df):
```

For each DataFrame row, the function:

1. canonicalizes every cell;
2. puts the canonical values into a `frozenset`; and
3. returns a list containing one set per row.

This representation is used only when the two results have different column
counts. Because a set has no column positions, a smaller row can be checked
against a larger row regardless of which column contains a value.

Example:

```text
smaller row: ["YS Chahal"]
larger row:  ["YS Chahal", 1391]

{"str:YS Chahal"} is a subset of
{"str:YS Chahal", "num:1391"}
```

### 4. Convert rows to tuples with `_rows_as_tuples`

```python
def _rows_as_tuples(df, order_sensitive):
```

For results with the same number of columns, every row becomes a tuple of
canonical values. Tuples preserve column position.

- If `order_sensitive=True`, the rows remain in their original order.
- If `order_sensitive=False`, the rows are sorted before comparison.

Consequently, column names are ignored, but column positions are significant
when both DataFrames have the same number of columns.

### 5. Copy both DataFrames

At the start of `compare_dataframes`:

```python
gold_df = gold_df.copy()
generated_df = generated_df.copy()
```

This avoids accidentally mutating DataFrames owned by the calling code. The
current comparison logic does not modify them, but copying provides isolation
for future changes.

### 6. Require the same number of rows

```python
if len(gold_df) != len(generated_df):
    return False
```

A generated result with an extra or missing row is always incorrect. Column
leniency does not apply to row count.

### 7. Compare results with the same column count

```python
if gcols == xcols:
    return _rows_as_tuples(...) == _rows_as_tuples(...)
```

Both results are canonicalized and compared as lists of tuples.

For example, these match when row order is ignored:

```text
gold:      [("A", 10), ("B", 20)]
generated: [("B", 20.0), ("A", "10")]
```

They match because numeric representations are normalized and both row lists
are sorted.

### 8. Reject a column-count difference greater than one

```python
if abs(gcols - xcols) != 1:
    return False
```

Only one extra or missing column is tolerated. For example, a one-column gold
result cannot match a three-column generated result.

### 9. Identify the smaller and larger result

```python
small, large = ...
```

The logic is symmetric: either the gold or generated result may have the extra
column. The DataFrame with fewer columns becomes `small`.

### 10. Align rows and perform subset comparison

Both DataFrames are converted to lists of row sets. When order is not
significant, those lists are sorted using each set's sorted values as the key.
The evaluator then checks:

```python
all(s.issubset(l) for s, l in zip(small_rows, large_rows))
```

Every value in each smaller row must occur in its corresponding larger row.
This accepts both of the following:

```text
gold:      ["YS Chahal", 1391]
generated: ["YS Chahal"]
```

```text
gold:      ["YS Chahal"]
generated: ["YS Chahal", 1391]
```

### 11. Return a structured result with `evaluate_one`

`evaluate_one` is the public wrapper normally called by the orchestration code.

It returns one of these dictionaries:

```python
{"correct": True,  "reason": "match"}
{"correct": False, "reason": "mismatch"}
{"correct": False, "reason": "sql_error"}
{"correct": False, "reason": "compare_error: ..."}
```

The cases mean:

- `match`: the DataFrames passed the comparison;
- `mismatch`: comparison completed, but values or shapes did not match;
- `sql_error`: the caller supplied `None`, normally because generated SQL could
  not be executed;
- `compare_error`: an unexpected exception occurred during comparison.

## Example usage

```python
import pandas as pd
from evaluator import evaluate_one

gold_df = pd.DataFrame(
    [["YS Chahal", 1391]],
    columns=["bowler", "legal_balls"],
)

generated_df = pd.DataFrame(
    [["YS Chahal"]],
    columns=["bowler"],
)

result = evaluate_one(gold_df, generated_df)
print(result)
# {'correct': True, 'reason': 'match'}
```

For a question whose answer must be ordered, pass the dataset's
`order_sensitive` flag:

```python
result = evaluate_one(
    gold_df,
    generated_df,
    order_sensitive=True,
)
```

In `golden_hard.csv`, question 12 is an example where this should be `True`
because the requested output must be ordered by season.

## Important limitations

1. **SQL meaning is not evaluated.** A logically wrong query passes if its
   returned values happen to match the gold result on this database.
2. **Text comparison is case-sensitive.** `"chahal"` and `"Chahal"` do not
   match.
3. **Whitespace is ignored only at the ends.** Internal whitespace is preserved.
4. **Four-decimal rounding can hide small numeric errors.** This is intentional
   tolerance, but it is not a relative-error comparison.
5. **Set conversion discards duplicate cells within a row.** For unequal column
   counts, `[1, 1]` becomes the same set as `[1]`.
6. **One missing column is accepted in either direction.** A generated result can
   omit one gold column if its remaining values form the required subset. This
   is deliberately lenient and may accept an under-specified answer.
7. **Column order matters for equal-width results.** Column names are ignored,
   but swapped values fail unless the values happen to be identical.
8. **Duplicate identical rows are retained.** Rows are stored in a list, so row
   multiplicity is preserved even though values inside unequal-width rows use
   sets.

## Summary

The evaluator implements execution-result accuracy with controlled leniency:

- values are normalized before comparison;
- column names are ignored;
- row order is optional;
- row counts must match exactly;
- equal-width results are compared positionally; and
- results differing by one column use row-wise subset matching.

This makes it practical for text-to-SQL evaluation where different SQL queries
can return the same essential answer in slightly different shapes.
