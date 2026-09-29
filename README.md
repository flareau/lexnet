# Lexnet

Utilities to load Lexical Network export files into Pandas DataFrames and Python objects.

## Requirements

- Python 3.10+
- `pandas`

## Project Layout

- `loader.py`: loads data into dataframes
- `models.py`: domain classes
- `lexnet.py`: wrapper
- `tests/`: unit and integration tests

## Basic Usage

```python
import lexnet as ln

fr = ln.LexicalNetwork("/path/to/data")
```

You can also load raw tables directly:

```python
from loader import load

tables = load("/path/to/data")
```

## CLI Smoke Test

```bash
python lexnet.py -d /path/to/data
```

Optional flags:

- `-s` / `--separator` (default: tab)
- `-e` / `--encoding` (default: `utf8`)

## Desktop Explorer

Launch the lightweight read-only browser GUI with a data path:

```bash
python explorer.py /path/to/data
```

The explorer opens a private local URL in your default browser. It supports
word and inflected-form lookup, lexical-function searches, and filtering
lexical entries by grammatical feature. Press Ctrl-C in the terminal to stop
it.

## Tests

Run all tests:

```bash
python -m unittest discover -s tests -v
```

`tests/test_data.py` is data-dependent and skips unless environment variable `LEXNET_DATA_PATH` is set.

Run with real data:

```bash
LEXNET_DATA_PATH="/absolute/path/to/data" python -m unittest discover -s tests -v
```
