import argparse
import csv
import pkg_resources
import spacy
import sys
import yaml

from .find_quotes import read_docs, setup_logging

# FIXME avoid code duplication with find_quotes


DEFAULT_RULES_FILE = 'rules_actors.yaml'


def extract_name(token):

    def _extract_deps(token):
        result = [token.lemma_]
        for t in token.children:
            if t.dep_ == 'flat:name':
                result.extend(_extract_deps(t))
        return result

    return ' '.join(_extract_deps(token))


def extract_organisation(token):
    result = [token.lemma_]
    for t in token.children:
        if t.dep_ == 'nmod:poss':
            result = [t2.orth_ for t2 in t.subtree] + result
    return ' '.join(result)


def find_actors(matcher, docs):
    for d in docs:
        for m_id, toks in matcher(d):
            try:
                name, role, org, lname = None, None, None, None
                pat_id = d.vocab[m_id].orth_
                if pat_id == 'actor-1':
                    name, org, role, lname = toks[0], toks[2], toks[1], toks[3]
                if pat_id == 'actor-2':
                    name, role, org, lname = toks[1], toks[0], toks[2], toks[3]
                yield {
                    'articleId': d.user_data['articleId'],
                    'sentenceId': d.user_data['sentenceId'][d[name].i],
                    'wordId': d.user_data['wordId'][d[name].i],
                    'name': extract_name(d[name]),
                    'role': d[role].lemma_,
                    'organisation': extract_organisation(d[org]),
                }
            except Exception as e:
                logging.warning(
                    'Exception in find_actors() - articleId={}, sentenceId={}: {}'\
                    .format(doc.user_data['articleId'],
                            doc.user_data['sentenceId'][toks[0]],
                            str(e)))


def load_rules(filename):
    if filename is not None:
        with open(filename) as fp:
            return yaml.safe_load(fp)
    else:
        data = pkg_resources.resource_string(__name__, DEFAULT_RULES_FILE)
        return yaml.safe_load(data)


def write_results(results, filename):
    fieldnames = ('articleId', 'sentenceId', 'wordId',
                  'name', 'organisation', 'role')

    if not filename or filename == '-':
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    else:
        with open(filename, 'w+') as outfp:
            writer = csv.DictWriter(outfp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Rule-based quote detection.')
    parser.add_argument('-i', '--input-file', metavar='FILE')
    parser.add_argument('-r', '--rules-file', metavar='FILE')
    parser.add_argument('-o', '--output-file', metavar='FILE')
    parser.add_argument('--logfile', metavar='FILE')
    parser.add_argument('-L', '--logging-level', metavar='LEVEL',
                        default='WARNING',
                        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    return parser.parse_args()


# TODO this code is almost identical to find_quotes.main()
# -> isolate the commonalities?
def main():
    args = parse_arguments()
    setup_logging(args.logfile, args.logging_level)
    rules = load_rules(args.rules_file)
    nlp = spacy.blank('fi')
    matcher = spacy.matcher.DependencyMatcher(nlp.vocab)
    for pat_id, pattern in rules['PATTERNS'].items():
        matcher.add(pat_id, pattern)

    with open(args.input_file) as infp:
        docs = read_docs(infp, nlp.vocab)
        results = find_actors(matcher, docs)
        write_results(results, args.output_file)

