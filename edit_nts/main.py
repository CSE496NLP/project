import os

import spacy

import dataset

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])


# Adapted from https://github.com/yuedongP/EditNTS/blob/master/data_preprocess.py
# Currently don't have a use for this
def remove_lrb(sent_string):
    frac_list = sent_string.split('-LRB-')
    clean_list = []
    for phrase in frac_list:
        if '-rrb-' in phrase:
            clean_list.append(phrase.split('-RRB-')[1].strip())
        else:
            clean_list.append(phrase.strip())
    clean_sent_string = ' '.join(clean_list)
    return clean_sent_string


# Currently just feeds the complex and simple sentences into spaCy.
# There probably isn't a good reason to do this for the simplified text.
def process_raw_lines(raw_comp_lines, raw_simp_lines):
    comp_tokens = list(nlp.pipe(raw_comp_lines))
    simp_tokens = list(nlp.pipe(raw_simp_lines))

    return {
        'comp_tokens': comp_tokens,
        'simp_tokens': simp_tokens
    }


def main():
#    print('Loading wikilarge')
#    comp_lines, simp_lines = dataset.wikilarge()

#    print('Processing raw data')
#    tokens = process_raw_lines(comp_lines, simp_lines)

#    print('Processed {} tokens'.format(len(tokens['comp_tokens'])))

    print('Loading wikiSmall')
    comp_lines, simp_lines = dataset.wikismall()

    print('Processing raw data')
    tokens = process_raw_lines(comp_lines, simp_lines)

    print('Processed {} tokens'.format(len(tokens['comp_tokens'])))


if __name__ == "__main__":
    main()
