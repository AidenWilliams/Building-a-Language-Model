from lxml import etree
import re
from tqdm.notebook import tqdm
import os
from collections import defaultdict
import numpy as np


class Corpus(object):
    def __init__(self, corpus=None, directory='Corpus/', verbose=False):
        if corpus is None:
            self._corpus = self.CorpusAsListOfSentences(directory, verbose)
        else:
            self._corpus = corpus
        identifier = tuple([1, 'vanilla'])
        self._ngrams = {}
        self._ngrams[identifier] = self.NGram(n=1, verbose=verbose)
        self._models = {}
        self._models[identifier] = self.Model(n=1, verbose=verbose)

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
        # Remove any double spaces
        word = np.char.replace(word, "  ", " ")

        word = str(np.char.lower(word))
        if word == "" or word == " ":
            return None
        return word

    @staticmethod
    def CorpusAsListOfSentences(root='Corpus/', verbose=False):
        roots = Corpus._ParseAsXML(root, verbose)
        sentences = []
        for root in tqdm(roots, desc='XML File', disable=not verbose):
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

    def Counts(self, n, verbose=False):
        counts = defaultdict(lambda: 0)
        for s in tqdm(self, desc='Counting x counts', disable=not verbose):
            for i in range(len(s) + 1):
                if i < n:
                    continue
                count = []
                for x in range(n, 0, -1):
                    count.append(s[i - x])
                count = tuple(count)

                counts[count] += 1

        return counts

    def NGram(self, n=2, model='vanilla', verbose=False):
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        identifier = tuple([n, model])
        if identifier in self._ngrams:
            if self._ngrams[identifier]['model'] == model:
                return self._ngrams[identifier]

        if model == 'laplace' or model == 'vanilla':
            counts = self.Counts(n, verbose)

            if model == 'laplace':
                for x in counts:
                    counts[x] += 1

        else:
            _count = self.Counts(1, verbose)
            tc = []
            for s in self:
                ts = []
                for w in s:
                    if _count[tuple([w])] < 3:
                        ts.append('UNK')
                    else:
                        ts.append(w)
                tc.append(ts)

            temp = Corpus(corpus=tc, verbose=verbose)
            counts = temp.Counts(n, verbose)

        result = {
            'count': counts,
            'model': model
        }

        self._ngrams[identifier] = result
        return self._ngrams[identifier]

    def GetCount(self, sequence: tuple, model='vanilla', verbose=False):
        n = len(sequence)
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        _ngram = self.NGram(n, model, verbose)['count']

        if sequence in _ngram:
            return _ngram[sequence]
        else:
            if model == 'laplace':
                return 1
            else:
                return 0

    def Model(self, n=2, model='vanilla', verbose=False):
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        identifier = tuple([n, model])
        if identifier in self._models:
            if self._models[identifier].model == model:
                return self._models[identifier]

        self._models[identifier] = Model(corpus=self, n=n, model=model, verbose=verbose)
        return self._models[identifier]

    def GetProbability(self, input, n, model='vanilla', verbose=False):
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        if type(input) is str:
            tc = Corpus([[input]], verbose=verbose)
        else:
            paragraph = False
            for elem in input:
                if isinstance(elem, list):
                    paragraph = True
                if paragraph and not isinstance(elem, list):
                    raise Exception('Input must be of the forms:\nstr\n[str, str, str]\n[[str, str, str], ...,  [str, '
                                    'str, str]].')

            if paragraph:
                tc = Corpus(input)
            else:
                tc = Corpus([input])

        _model = self.Model(n=n, model=model, verbose=verbose)
        _ngram = tc.NGram(n=n, model=model, verbose=verbose)

        input_probability = 1
        exists = False
        for _n in _ngram['count']:
            exists = True
            input_probability *= _model.GetProbabilityMath(_n[-1], _n[:n - 1]) ** tc.GetCount(_n, model=model)

        if not exists:
            input_probability = 0

        return input_probability

    def LinearInterpolation(self, trigram, model='vanilla', verbose=False):
        if len(trigram) != 3 or type(trigram) != type(tuple):
            raise Exception('trigram input must be a tuple of 3 words.')

        l1 = 0.1
        l2 = 0.3
        l3 = 0.6

        return l3 * self.GetProbability(input=[trigram[2], trigram[0], trigram[1]], n=3, model=model, verbose=verbose) + \
               l2 * self.GetProbability(input=[trigram[2], trigram[1]], n=2, model=model, verbose=verbose) + \
               l1 * self.GetProbability(input=trigram[2], n=1, model=model, verbose=verbose)


class Model(object):
    def __init__(self, corpus, n=2, model='vanilla', verbose=False):
        V = 0
        cmodel = model
        if model == 'laplace':
            cmodel = 'vanilla'
            V = len(corpus.NGram(n=1, verbose=verbose)['count'])

        counts = corpus.NGram(n, model=cmodel, verbose=verbose)['count']

        probabilities = {}
        self.N = len([w for s in corpus for w in s])

        if n is not 1:
            previous = corpus.NGram(n - 1, model=cmodel, verbose=verbose)['count']
            for x in counts:
                probabilities[x] = {
                    'probability': (counts[x] + int(model == 'laplace')) / (previous[x[:n - 1]] + V)}
        else:
            for x in counts:
                probabilities[x] = {'probability': (counts[x] + int(model == 'laplace')) / (self.N + V)}
        self.probabilities = probabilities
        self.model = model

    # ('z', tuple(x, y))
    def GetProbabilityMath(self, forX, givenY:tuple):
        sequence = givenY + (forX,)

        if sequence in self.probabilities:
            return self.probabilities[sequence]['probability']
        else:
            return 0

    def Perplexity(self):
        prob = 1
        for p in self.probabilities:
            prob *= self.probabilities[p]['probability']
        if prob == 0:
            return 0
        else:
            return prob ** -(1 / self.N)


