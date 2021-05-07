from lxml import etree
import re
from tqdm.notebook import tqdm
import os
import numpy as np
from typing import Union, List


class Corpus(object):
    """
    List[List[str]] type class with static functionality to read data from the Malti dataset, xml files

    Each xml file is split in <text> tags, each <text> tag is split in <p> tags, each <p> is split in <s> tags and
    finally each new line is considered a word, tab separated in the following format:

            word, speech tag, lemma and morphological root.

    This class handles the reading and parsing of files as described above, and is intended to be used as a List[List[str]]
    with added functionality.

    Attributes
    ----------
    _corpus : List[List[str]]
        a list of sentences, each being a list of words
    """

    def __init__(self, corpus=Union[List[List[str]], None], directory='Corpus/', verbose=False):
        """
        Initialises the corpus object, corpus can be copy constructed from any other List[List[str]] variable,
        otherwise it will be build using the class' functionality.
        """
        if corpus is None:
            self._corpus = self.CorpusAsListOfSentences(directory, verbose)
        else:
            self._corpus = corpus

    def __len__(self):
        """
        returns length of _corpus
        """
        return len(self._corpus)

    def __iter__(self):
        """
        gives functionality to iterate over _corpus
        """
        for word in self._corpus:
            yield word

    def __getitem__(self, index):
        """
        returns item in _corpus at index
        """
        return self._corpus[index]

    # Create functions to free memory once function scope is left
    @staticmethod
    def _ReadCorpus(root='Corpus/', verbose=False):
        """
        This function checks the root location of where all xml files are contained and if it encounters no issue
        accessing it, it will read the contents of the files within it and return them in the form of a list.
        """
        if not os.access(root, os.R_OK):
            print('Check root!!')

        xml_data = []

        for file in tqdm(os.listdir(root), desc='Reading Files', disable=not verbose):
            xml_data.append(open(os.path.join(root, file), 'r', encoding='utf8').read())  # Read file
        return xml_data

    @staticmethod
    def _ParseAsXML(root='Corpus/', verbose=False):
        """
        The parser being used is initialised and the xml data from the files is read into xml_data. Each file is
        then parsed and appended to roots. Each file is split in a number of texts, so roots is a list of
        these parsed texts.
        """
        parser = etree.XMLParser(recover=True)
        roots = []
        xml_data = Corpus._ReadCorpus(root, verbose)
        for xml in tqdm(xml_data, desc='Parsing XML', disable=not verbose):
            roots.append(etree.fromstring(xml, parser=parser))
        return roots

    @staticmethod
    def filterFurther(word: str):
        """
        Filters a given word from spaces, symbols and removes capitalization.
        """
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
        """
        Creates the List[List[str]] _corpus object by reading and parsing the xml files. Each sentence is filtered to
        get the words. Then each word found is filtered by only looking at the word value and then filtered using
        filterFurther appended to the sublist. The <s> and </s> tokens are inserted in their place and considered as
        words.
        """
        roots = Corpus._ParseAsXML(root, verbose)
        sentences = []
        for root in tqdm(roots, desc='Building Sentences', disable=not verbose):
            for p in tqdm(root, desc='Paragraph', disable=not verbose):
                for s in p:
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