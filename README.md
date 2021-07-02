# Rule-based quote detection for Finnish news

## Installation

```
python3 setup.py install
```

## Usage

```
find_quotes -i [INPUT_FILE] -r [RULES_FILE] -o [OUTPUT_FILE]
```

* `INPUT_FILE` is the set of articles in CoNLL-CSV (tidytext) format,
* `RULES_FILE` is a YAML file containing the rules
  (default: `src/flopo_quote_detection/rules.yaml`)
* `OUTPUT_FILE` is the CSV file that will contain the results; if none is
  given, the results are printed to stdout.

The minimal working call is thus:
```
find_quotes -i [INPUT_FILE]
```
