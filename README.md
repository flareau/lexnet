# Lexnet

Utilities to load Lexical Network export files into Pandas DataFrames and Python objects.

## Requirements

- Python 3.10+
- `pandas`

## Project Layout

- `loader.py`: loads data into dataframes
- `models.py`: domain classes
- `lexnet.py`: wrapper
- `explorer.py`: read-only browser interface
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

## Browser Explorer

Launch the lightweight read-only interface with a LexNet export folder:

```bash
python explorer.py /path/to/data
```

The explorer starts a server bound to `127.0.0.1` and opens its private local
URL in the default browser. It does not require Tkinter or a web framework.
Press Ctrl-C in the terminal to stop it.

Optional flags:

- `--port PORT`: use a specific local port instead of an automatically selected one
- `--no-browser`: start the server without opening the browser automatically

### Search

- **Words:** search entry names, lexical-unit names, and optionally inflected
  forms using exact, prefix, or substring matching.
- **Lexical functions:** enter one or more LF names, or browse the LF model by
  group, family, and function. Group headings are selectable and search the
  entire group; selecting a family can search the whole family or one LF.
- **Features:** find entries and lexical units with one or more grammatical
  features, using either match-any or match-all behavior.

Result columns can be sorted by clicking their headings. Entry and
lexical-unit names are links to the Inspector.

### Inspector

The Inspector displays lexical entries and lexical units. Entry views list
their lexical units with compact grammatical summaries. Lexical-unit views
include grammatical information, definitions, wordforms, semantic labels,
propositional forms, lexical functions, and examples when available. Related
entries and lexical units are navigable links.

Internal LexNet IDs are used for navigation but omitted from the display.
Names for grammatical features, wordform features, semantic labels, and
lexical functions are resolved from their XML model files.

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
