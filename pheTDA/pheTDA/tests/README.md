# Repository checks

Use Python 3.9 or newer.

Install the pipeline and development tools from the repository root:

```bash
python -m pip install -r requirements-dev.txt
```

Run the behavioral test suite:

```bash
python -m pytest
```

Run static lint checks without creating a cache directory:

```bash
python -m ruff check --no-cache .
```

Verify formatting without modifying files:

```bash
python -m ruff format --check .
```

Pytest verifies runtime behavior. Ruff checks source quality and formatting;
the two checks are complementary.
