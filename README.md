# Rule-based processing of Finnish news

This repository contains the rule-based quote detection engine described
in the following publication:

Maciej Janicki, Antti Kanner and Eetu Mäkelä.
[Detection and attribution of quotes in Finnish news media: BERT vs. rule-based approach](https://openreview.net/forum?id=YTVwaoG0Mi).
In: *Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)*,
Tórshavn, Faroe Islands, May 2023.

## Installation

```
python3 setup.py install
```

## Quote detection

A rule-based method for quote detection described in the following article:

Maciej Janicki, Antti Kanner and Eetu Mäkelä.
[Detection and attribution of quotes in Finnish news media: BERT vs. rule-based approach](https://openreview.net/forum?id=YTVwaoG0Mi).
In: *Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)*,
Tórshavn, Faroe Islands, May 2023.

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

The input file should be in CoNLL-CSV format. Example input and ground
truth files can be found in
[hsci-r/fi-quote-coref-corpus](https://github.com/hsci-r/fi-quote-coref-corpus) 
and the `flopo-eval` tool from
[flopo-formats](https://github.com/hsci-r/flopo-formats) can be used
for evaluation.

## Actor detection

```
find_actors -i [INPUT_FILE] -r [RULES_FILE] -o [OUTPUT_FILE]
```

Arguments are the same as for `find_quotes`.
