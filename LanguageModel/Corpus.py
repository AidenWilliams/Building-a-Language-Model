from lxml import etree
import re
from tqdm.notebook import tqdm
import os
import numpy as np
from typing import Union, List, Any


class Corpus(object):
    def __init__(self, corpus=Union[List[List[Union[str, Any]]], None], directory='Corpus/', verbose=False):
        if corpus is None:
            self._corpus = self.CorpusAsListOfSentences(directory, verbose)
        else:
            self._corpus = corpus

    def __len__(self):
        return len(self._corpus)

    def __iter__(self):
        for word in self._corpus:
            yield word

    def __getitem__(self, index):
        return self._corpus[index]

    # Create functions to free memory once function scope is left
    @staticmethod
    def _ReadCorpus(root='Corpus/', verbose=False):
        if not os.access(root, os.R_OK):
            print('Check root!!')

        xml_data = []

        for file in tqdm(os.listdir(root), desc='Reading Files', disable=not verbose):
            xml_data.append(open(os.path.join(root, file), 'r', encoding='utf8').read())  # Read file
        return xml_data

    @staticmethod
    def _ParseAsXML(root='Corpus/', verbose=False):
        parser = etree.XMLParser(recover=True)
        roots = []
        xml_data = Corpus._ReadCorpus(root, verbose)
        for xml in tqdm(xml_data, desc='Parsing XML', disable=not verbose):
            roots.append(etree.fromstring(xml, parser=parser))
        return roots

    @staticmethod
    def filterFurther(word: str):
        # remove symbols
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n,"
        for i in range(len(symbols)):
            word = np.char.replace(word, symbols[i], ' ')
        # remove apostrophe
        word = np.char.replace(word, "'", "")
        # Remove any spaces
        word = np.char.replace(word, " ", "")

        word = str(np.char.lower(word))
        if word == "" or word == " ":
            return None
        return word

    @staticmethod
    def CorpusAsListOfSentences(root='Corpus/', verbose=False):
        roots = Corpus._ParseAsXML(root, verbose)
        sentences = []
        for root in tqdm(roots, desc='Building Sentences', disable=not verbose):
            for i, p in tqdm(enumerate(root), desc='Paragraph', disable=not verbose):
                for k, s in enumerate(p):
                    unfiltered_sentence = re.split(r'\n', s.text.lstrip('\n'))
                    sentence = []
                    for unfiltered_word in unfiltered_sentence:
                        if unfiltered_word is not '':
                            filtered_word = unfiltered_word.split('\t')
                            full_filter = Corpus.filterFurther(filtered_word[0])
                            if full_filter is not None:
                                sentence.append(full_filter)

                    if sentence is not []:
                        sentence.insert(0, '<s>')
                        sentences.append(sentence)
                        sentence.append('</s>')
        return sentences
