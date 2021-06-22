import argparse
import csv
from collections import defaultdict
import spacy
import yaml
import warnings

# TODO
# - deal with errors while reading the data
#   - ignore documents that produce errors, but continue
# - add the 4 rules used in the Prolog code
# - quotation marks and multi-sentence quotes
# - author name extraction from the head
# - naive pronoun resolution
# - message nouns


def read_docs(fp, vocab):
    'Read documents from CoNLL-CSV format to spaCy Doc objects.'
    reader = csv.DictReader(fp)
    spacy.tokens.Token.set_extension('sentenceId', default=None)
    spacy.tokens.Token.set_extension('wordId', default=None)
    
    cur_doc_id, pos, offset, sent_ids, tok_ids = None, 0, -1, [], []
    tokens = defaultdict(lambda: list())
    for row in reader:
        if row['articleId'] != cur_doc_id:
            if cur_doc_id is not None:
                yield spacy.tokens.Doc(
                    vocab,
                    user_data={ 'articleId': cur_doc_id,
                                'sentenceId': sent_ids,
                                'wordId': tok_ids },
                    **tokens)
                sent_ids, tok_ids = [], []
                tokens = defaultdict(lambda: list())
            cur_doc_id, pos, offset = row['articleId'], 0, -1
        if int(row['wordId']) == 1:
            offset = pos-1
        if not row['word']:
            warnings.warn(\
                'Ignoring empty token: articleId={} sentenceId={} wordId={}'\
                .format(row['articleId'], row['sentenceId'], row['wordId']))
            pos += 1
            continue
        tokens['words'].append(row['word'])
        tokens['lemmas'].append(row['lemma'])
        tokens['pos'].append(row['upos'])
        tokens['morphs'].append(row['feats'])
        # In CoNLL-CSV the root's head is set to 0, but in spaCy
        # it must be set to the token itself.
        head = int(row['head'])
        tokens['heads'].append(pos if head == 0 else head+offset)
        tokens['deps'].append(row['deprel'])
        tokens['spaces'].append('SpaceAfter=No' not in row['misc'])
        sent_ids.append(row['sentenceId'])
        tok_ids.append(row['wordId'])
        pos += 1
    yield spacy.tokens.Doc(
        vocab,
        user_data={ 'articleId': cur_doc_id,
                    'sentenceId': sent_ids,
                    'wordId': tok_ids },
        **tokens)


def load_yaml(filename):
    with open(filename) as fp:
        return yaml.load(fp)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Rule-based quote detection.')
    parser.add_argument('-i', '--input-file', metavar='FILE')
    parser.add_argument('-r', '--rules-file', default='rules.yaml', metavar='FILE')
    parser.add_argument('-o', '--output-file', metavar='FILE')
    return parser.parse_args()

    
def main():
    args = parse_arguments()
    rules = load_yaml(args.rules_file)
    nlp = spacy.blank('fi')
    matcher = spacy.matcher.DependencyMatcher(nlp.vocab)
    matcher.add('quote_triplet', rules['PATTERNS'])

    with open(args.output_file, 'w+') as outfp:
        writer = csv.DictWriter(
            outfp,
            fieldnames=('articleId', 'startSentenceId', 'startWordId',
                        'endSentenceId', 'endWordId', 'author',
                        'authorSentenceId', 'authorWordId', 'direct'))
        writer.writeheader()
        with open(args.input_file) as infp:
            for d in read_docs(infp, nlp.vocab):
                for m_id, toks in matcher(d):
                    prop = list(d[toks[2]].subtree)
                    writer.writerow({
                        'articleId': d.user_data['articleId'],
                        'startSentenceId': d.user_data['sentenceId'][prop[0].i],
                        'startWordId': d.user_data['wordId'][prop[0].i],
                        'endSentenceId': d.user_data['sentenceId'][prop[-1].i],
                        'endWordId': d.user_data['wordId'][prop[-1].i],
                        'author': d[toks[1]],
                        'authorSentenceId': d.user_data['sentenceId'][toks[1]],
                        'authorWordId': d.user_data['wordId'][toks[1]],
                        'direct': 'false',
                    })

if __name__ == '__main__':
    main()

