import argparse
import csv
from collections import defaultdict
import spacy
import sys
import yaml
import warnings

# TODO
# - naive pronoun resolution
# - integrate NER into author detection? (capture complete names, esp. of
#   organizations)

def extract_author(token, lexicon):
    # message nouns (constructions like "Xn ehdotuksen/tietojen/tulkinnan mukaan")
    if token.lemma_ in lexicon['MESSAGE_NOUNS']:
        for d in token.children:
            if d.dep_ in ('nmod:poss', 'nmod:gsubj'):
                token = d
                break
    # constructions of the form "puheenjohtajan Antti Palolan mukaan"
    # puheenjohtaja[NOUN] --[appos]--> Antti[PROPN] --[flat:name]--> Palola[PROPN]
    if token.pos_ == 'NOUN':
        for d in token.children:
            if d.pos_ == 'PROPN' and d.dep_ == 'appos':
                token = d
                break
    # names in form "FirstName LastName"
    for d in token.children:
        if d.dep_ == 'flat:name':
            return ' '.join([token.lemma_, d.lemma_])
    return token.lemma_


def extract_proposition(doc, cue, prop_head, pat_id):

    def _find_par_start(doc, token):
        '''Finds the token at the beginning of the paragraph
           starting at `token`.'''
        par_id = doc.user_data['paragraphId'][token.i]
        par_start = token
        while par_start.i > 0 and \
                doc.user_data['paragraphId'][par_start.i-1] == par_id:
            par_start = doc[par_start.i-1]
        return par_start

    def _quote_between(doc, tok_1, tok_2):
        i = min(tok_1.i, tok_2.i)
        while i < max(tok_1.i, tok_2.i):
            if doc[i].norm_ == '"':
                return i
            i += 1
        return None
    
    def _find_matching_quote(doc, start, direction):
        i = start+direction
        while i > 0 and i < len(doc):
            if doc[i].norm_ == '"':
                return i
            i += direction
        return None
    
    tokens = [t for t in prop_head.subtree \
                    if (t not in cue.subtree and t.dep_ != 'punct') \
                       or prop_head.head == cue]
    if not tokens:
        raise Exception('No proposition extracted')
    start, end = tokens[0], tokens[-1]
    direct = False

    # if the paragraph starts with hyphen -> extend the proposition until there
    if pat_id == 2:
        par_start = _find_par_start(doc, prop_head)
        if par_start.norm_ == '-':
            start = par_start
            direct = True

    # there is a quotation mark between cue and proposition
    # -> proposition is enclosed in quotes
    q = _quote_between(doc, cue, prop_head)
    if q is not None:
        direction = 1 if prop_head.i > cue.i else -1
        q2 = _find_matching_quote(doc, q, direction)
        if q2 is not None and prop_head.i in range(min(q, q2), max(q, q2)):
            start = doc[min(q, q2)]
            end = doc[max(q, q2)]
            direct = True
    
    return (doc.user_data['sentenceId'][start.i],
            doc.user_data['wordId'][start.i],
            doc.user_data['sentenceId'][end.i],
            doc.user_data['wordId'][end.i],
            direct)


def find_matches(matcher, docs, lexicon):
    for d in docs:
        for m_id, toks in matcher(d):
            prop_s_start, prop_w_start, prop_s_end, prop_w_end, direct = \
                (None, None, None, None, None)
            try:
                prop_s_start, prop_w_start, prop_s_end, prop_w_end, direct = \
                    extract_proposition(d, d[toks[0]], d[toks[2]], m_id)
            except Exception as e:
                warnings.warn(e)
            yield {
                'articleId': d.user_data['articleId'],
                'startSentenceId': prop_s_start,
                'startWordId': prop_w_start,
                'endSentenceId': prop_s_end,
                'endWordId': prop_w_end,
                'author': extract_author(d[toks[1]], lexicon),
                #'cue': d[toks[0]],
                'authorHead': d.user_data['sentenceId'][toks[1]] + '-' \
                              + d.user_data['wordId'][toks[1]],
                #'authorSentenceId': d.user_data['sentenceId'][toks[1]],
                #'authorWordId': d.user_data['wordId'][toks[1]],
                #'pattern': m_id,
                'direct': 'true' if direct else 'false'
            }


def read_docs(fp, vocab):
    'Read documents from CoNLL-CSV format to spaCy Doc objects.'

    def make_doc(vocab, tokens, doc_id, par_ids, sent_ids, tok_ids):
        try:
            return spacy.tokens.Doc(
                vocab,
                user_data={ 'articleId': cur_doc_id,
                            'paragraphId': par_ids,
                            'sentenceId': sent_ids,
                            'wordId': tok_ids },
                **tokens)
        except Exception as e:
            warnings.warn(
                'Ignoring articleId=\'{}\': There is something wrong'
                ' with the document - please investigate.'\
                .format(doc_id))
    
    reader = csv.DictReader(fp)
    spacy.tokens.Token.set_extension('paragraphId', default=None)
    spacy.tokens.Token.set_extension('sentenceId', default=None)
    spacy.tokens.Token.set_extension('wordId', default=None)
    
    cur_doc_id, pos, offset, par_ids, sent_ids, tok_ids = None, 0, -1, [], [], []
    tokens = defaultdict(lambda: list())
    for row in reader:
        if row['articleId'] != cur_doc_id:
            if cur_doc_id is not None:
                doc = make_doc(vocab, tokens, cur_doc_id, par_ids, sent_ids, tok_ids)
                if doc is not None:
                    yield doc
                par_ids, sent_ids, tok_ids = [], [], []
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
        par_ids.append(row['paragraphId'])
        sent_ids.append(row['sentenceId'])
        tok_ids.append(row['wordId'])
        pos += 1
    doc = make_doc(vocab, tokens, cur_doc_id, par_ids, sent_ids, tok_ids)
    if doc is not None:
        yield doc


def load_yaml(filename):
    with open(filename) as fp:
        return yaml.safe_load(fp)


def write_results(results, filename):
    fieldnames = ('articleId', 'startSentenceId', 'startWordId',
                  'endSentenceId', 'endWordId', 'author',
                  'authorHead', 'direct')

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
    parser.add_argument('-r', '--rules-file', default='rules.yaml', metavar='FILE')
    parser.add_argument('-o', '--output-file', metavar='FILE')
    return parser.parse_args()

    
def main():
    args = parse_arguments()
    rules = load_yaml(args.rules_file)
    nlp = spacy.blank('fi')
    matcher = spacy.matcher.DependencyMatcher(nlp.vocab)
    for pat_id, pattern in rules['PATTERNS'].items():
        matcher.add(pat_id, pattern)

    with open(args.input_file) as infp:
        docs = read_docs(infp, nlp.vocab)
        results = find_matches(matcher, docs, rules['LEXICON'])
        write_results(results, args.output_file)


if __name__ == '__main__':
    main()

