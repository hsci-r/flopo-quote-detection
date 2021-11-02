import argparse
import csv
from collections import defaultdict, namedtuple
import logging
from operator import itemgetter
import pkg_resources
import spacy
import sys
import yaml


DEFAULT_RULES_FILE = 'rules.yaml'

Author = namedtuple('Author', ['name'])
Quote = namedtuple('Quote', ['authors', 'proposition', 'direct', 'match'])
QuoteMatch = namedtuple('QuoteMatch',
                        ['cue', 'author_head', 'prop_head', 'pat_id'])

# TODO
# - naive pronoun resolution
# - integrate NER into author detection? (capture complete names, esp. of
#   organizations)
# - extract a noun phrase from author name
# - disambiguate author names
# - how to do resolution when there are multiple authors?
# - data structure "author data":
#   - first name, last name (as strings)
#   - title / role (noun phrase before the name)
# - extract_authors returns a list of authors
# - resolve_authors takes quotes from the entire document and resolves them
# - output:
#   - authorTitle (spaCy span)
#   - authorFirstName (spaCy span)
#   - authorLastName (spaCy span)
#   - authorHead (spaCy token)

def extract_authors(token, lexicon):
    return [extract_author(token, lexicon)] + \
           [extract_author(t, lexicon) for t in token.children \
                                       if t.dep_ == 'conj']


def extract_flat_name(token):
    result = [token]
    for d in token.children:
        if d.dep_ == 'flat:name':
            result.extend(extract_flat_name(d))
    return result


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
    return Author(name=extract_flat_name(token))


def author_to_str(a):
    return ' '.join(t.lemma_ for t in a.name)


def resolve_authors(doc, quotes, names):
    quotes_and_names = [(q.proposition[0].i, 'quote', q) for q in quotes] \
                       + [(n[0].i, 'name', n) for n in names]
    quotes_and_names.sort(key=itemgetter(0))
    names_by_last = {}
    for (i, t, x) in quotes_and_names:
        if t == 'name':
            names_by_last[x[-1].lemma_] = x
            names_by_last['hän'] = x
        elif t == 'quote':
            for a in x.authors:
                a_str = author_to_str(a)
                if a_str in names_by_last:
                    s_id = doc.user_data['sentenceId'][a.name[0].i]
                    del a.name[:]
                    a.name.extend(names_by_last[a_str])
                    names_by_last['hän'] = names_by_last[a_str]
                    logging.info(
                        'Resolving \'{}\' to \'{}\' in articleId={},'
                        ' sentenceId={}' \
                        .format(a_str, author_to_str(a),
                                doc.user_data['articleId'], s_id))


def extract_proposition(doc, match):

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
    
    tokens = [t for t in match.prop_head.subtree \
                    if (t not in match.cue.subtree and t.dep_ != 'punct') \
                       or match.prop_head.head == match.cue]
    if not tokens:
        raise Exception('No proposition extracted')
    start, end = tokens[0], tokens[-1]
    direct = False

    # if the paragraph starts with hyphen -> extend the proposition until there
    if match.pat_id == 'quote-2':
        par_start = _find_par_start(doc, match.prop_head)
        if par_start.norm_ == '-':
            start = par_start
            direct = True

    # there is a quotation mark between cue and proposition
    # -> proposition is enclosed in quotes
    q = _quote_between(doc, match.cue, match.prop_head)
    if q is not None:
        direction = 1 if match.prop_head.i > match.cue.i else -1
        q2 = _find_matching_quote(doc, q, direction)
        if q2 is not None and match.prop_head.i in range(min(q, q2), max(q, q2)):
            start = doc[min(q, q2)]
            end = doc[max(q, q2)]
            direct = True

    return doc[start.i:end.i+1], direct


def find_matches(matcher, doc, lexicon):
    result = {'quotes': list(), 'names': list()}
    for m_id, toks in matcher(doc):
        try:
            pat_id = doc.vocab[m_id].orth_
            if pat_id.startswith('quote'):
                match = QuoteMatch(cue=doc[toks[0]], author_head=doc[toks[1]],
                                   prop_head=doc[toks[2]], pat_id=pat_id)
                prop, direct = extract_proposition(doc, match)
                authors = extract_authors(match.author_head, lexicon)
                if not prop:
                    logging.warning(\
                        'Empty proposition in articleId={}, sentenceId={}'\
                        .format(doc.user_data['articleId'],
                                doc.user_data['sentenceId'][toks[0]]))
                q = Quote(proposition=prop, direct=direct,
                          match=match, authors=authors)
                result['quotes'].append(q)
            elif pat_id.startswith('name'):
                result['names'].append(extract_flat_name(doc[toks[0]]))
        except Exception as e:
            logging.warning(
                'Exception in find_matches() - articleId={}, sentenceId={}: {}'\
                .format(doc.user_data['articleId'],
                        doc.user_data['sentenceId'][toks[0]],
                        str(e)))
    return result

# Recognize direct quotes encompassing an entire paragraph, without a cue.
# The conditions are as follows:
# - the last sentence of the previous paragraph contains
#   an already recognized quote,
# - the beginning of this paragraph is not already marked as a quote,
# - the paragraph starts with a hyphen or is enclosed in quotation marks.
def quotes_from_paragraphs(doc, quotes_from_matches):
    
    def _next_paragraph(doc, token):
        i = token.i
        prev_par_id = int(doc.user_data['paragraphId'][i])
        while int(doc.user_data['paragraphId'][i]) != prev_par_id+1:
            i += 1
            if i >= len(doc):
                return None
        j = i
        while int(doc.user_data['paragraphId'][j]) == prev_par_id+1:
            j += 1
            if j >= len(doc):
                break
        return doc[i:j]
    
    quote_tokens = set(tok.i for q in quotes_from_matches for tok in q.proposition)
    for q in quotes_from_matches:
        try:
            np = _next_paragraph(doc, q.proposition[-1])
            if np is not None \
                    and int(doc.user_data['sentenceId'][np[0].i]) \
                        == int(doc.user_data['sentenceId'][q.proposition[-1].i])+1 \
                    and (np[0].norm_ == '-' \
                         or np[0].norm_ == '"' and np[-1].norm_ == '"') \
                    and not any(tok.i in quote_tokens for tok in np):
                m = QuoteMatch(author_head=q.match.author_head, prop_head=None,
                               cue=np[0], pat_id='paragraph')
                yield Quote(authors=q.authors, proposition=np, direct=True, match=m)
        except Exception as e:
            logging.warning(
                'Exception in quotes_from_paragraphs() -'
                ' articleId={}, sentenceId={}: {}'\
                .format(doc.user_data['articleId'],
                        doc.user_data['sentenceId'][q.match.cue.i],
                        str(e)))


def quote_to_dict(q, doc):
    return {
        'articleId': doc.user_data['articleId'],
        'startSentenceId': doc.user_data['sentenceId'][q.proposition[0].i],
        'startWordId': doc.user_data['wordId'][q.proposition[0].i],
        'endSentenceId': doc.user_data['sentenceId'][q.proposition[-1].i],
        'endWordId': doc.user_data['wordId'][q.proposition[-1].i],
        'author': '|'.join(author_to_str(a) for a in q.authors),
        'authorHead': doc.user_data['sentenceId'][q.match.author_head.i] + '-' \
                       + doc.user_data['wordId'][q.match.author_head.i],
        'direct': 'true' if q.direct else 'false'
    }


def find_quotes(matcher, docs, lexicon, resolve=True):
    for d in docs:
        m = find_matches(matcher, d, lexicon)
        doc_quotes = m['quotes'] + list(quotes_from_paragraphs(d, m['quotes']))
        doc_quotes.sort(key=lambda q: q.proposition[0].i)
        if resolve and doc_quotes:
            resolve_authors(d, doc_quotes, m['names'])
        for q in doc_quotes:
            yield quote_to_dict(q, d)


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
            logging.warning(
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
            logging.warning(\
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


def load_rules(filename):
    if filename is not None:
        with open(filename) as fp:
            return yaml.safe_load(fp)
    else:
        data = pkg_resources.resource_string(__name__, DEFAULT_RULES_FILE)
        return yaml.safe_load(data)


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


def setup_logging(logfile, level):
    if logfile is None:
        logging.basicConfig(level=level)
    else:
        logging.basicConfig(filename=logfile, level=level)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Rule-based quote detection.')
    parser.add_argument('-i', '--input-file', metavar='FILE')
    parser.add_argument('-r', '--rules-file', metavar='FILE')
    parser.add_argument('-o', '--output-file', metavar='FILE')
    parser.add_argument('--no-resolve', action='store_true',
                        help='Do not resolve author names.')
    parser.add_argument('--logfile', metavar='FILE')
    parser.add_argument('-L', '--logging-level', metavar='LEVEL',
                        default='WARNING',
                        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    return parser.parse_args()

    
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
        results = find_quotes(matcher, docs, rules['LEXICON'],
                              resolve=not args.no_resolve)
        write_results(results, args.output_file)


if __name__ == '__main__':
    main()

