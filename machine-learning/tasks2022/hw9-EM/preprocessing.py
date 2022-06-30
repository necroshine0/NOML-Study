from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """

    file = open(filename, 'r', encoding="utf8")
    root = ET.fromstring(file.read().replace("&", "&amp;"))
    file.close()

    def get_tuples(text_splitted):
        tuples = []
        for pair in text_splitted:
            p = pair.split('-')
            tuples.append((int(p[0]), int(p[1])))
        return tuples

    sents, targets = [], []

    for child in root:

        eng, cz, sure, possbl = [], [], [], []

        for ch in child:
            txt = ch.text
            if txt is None:
                continue

            txt = txt.split()
            if ch.tag == 'english':
                eng = txt
            elif ch.tag == 'czech':
                cz = txt
            elif ch.tag == 'sure':
                sure = get_tuples(txt)
            elif ch.tag == 'possible':
                possbl = get_tuples(txt)

        sents.append(SentencePair(eng, cz))
        targets.append(LabeledAlignment(sure, possbl))

    return (sents, targets)


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """

    sources, targets = [], []
    s_counts, t_counts = Counter(), Counter()
    source_dict, target_dict = {}, {}

    for elem in sentence_pairs:
        sources.extend(elem.source)
        targets.extend(elem.target)

    s_counts = Counter(sources)
    t_counts = Counter(targets)

    if freq_cutoff is None:
        for ind, tkn in enumerate(s_counts):
            source_dict[tkn] = ind
        for ind, tkn in enumerate(t_counts):
            target_dict[tkn] = ind

    else:
        for ind, tkn in enumerate(s_counts.most_common(freq_cutoff)):
            source_dict[tkn[0]] = ind
        for ind, tkn in enumerate(t_counts.most_common(freq_cutoff)):
            target_dict[tkn[0]] = ind

    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """

    def strs_to_ints(content, relevant_dict):
        arr = []
        for word in content:
            if word in relevant_dict:
                arr.append(relevant_dict[word])
            else:
                return []
        return arr

    digitized = []
    for elem in sentence_pairs:

        cz, eng = None, None
        cz = strs_to_ints(elem.target, target_dict)
        if len(cz) != 0:
            eng = strs_to_ints(elem.source, source_dict)
            if len(eng) == 0:
                continue
        else:
            continue

        digitized.append(
            TokenizedSentencePair(np.array(eng, dtype='int32'), np.array(cz, dtype='int32'))
        )

    return digitized