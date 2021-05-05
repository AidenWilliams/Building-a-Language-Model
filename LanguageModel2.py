from lxml import etree
import re
from tqdm.notebook import tqdm
import os
from collections import defaultdict
import numpy as np
import random
from typing import Union






class asdasd(object):

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
                tc = Corpus(input, verbose=verbose)
            else:
                tc = Corpus([input], verbose=verbose)

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
        if len(trigram) != 3 or type(trigram) != tuple:
            raise Exception('trigram input must be a tuple of 3 words.')

        l1 = 0.1
        l2 = 0.3
        l3 = 0.6

        return l3 * self.GetProbability(input=[trigram[2], trigram[0], trigram[1]], n=3, model=model, verbose=verbose) + \
               l2 * self.GetProbability(input=[trigram[2], trigram[1]], n=2, model=model, verbose=verbose) + \
               l1 * self.GetProbability(input=trigram[2], n=1, model=model, verbose=verbose)

    def _getClosestTo(self, word, n=2, model='vanilla', verbose=False):
        if n == 1:
            raise Exception('unigrams are unsupported by this function.')

        _ngram = self.NGram(n=n, model=model, verbose=verbose)

        word = word if word == '<s>' else self.filterFurther(word)

        keys = [x for x in _ngram['count'].keys() if x[0] == word]

        probsforword = {}
        highestv = 0
        highestk = ''
        for k in keys:
            probsforword[k] = self.GetProbability(input=k, n=n, model=model, verbose=verbose)

            skipchance = random.randint(0, 9)

            if skipchance == 5:
                continue

            if probsforword[k] > highestv and k != '<s>':
                highestk = k
                highestv = probsforword[k]

        return highestk[1:]

    def GenerateSentence(self, startword='<s>', n=2, model='vanilla', verbose=False):
        sentence = []
        if startword != '<s>':
            sentence.append(startword)

        if n != 1:
            _model = self.Model(n=n, model=model, verbose=verbose)
            next = self._getClosestTo(word=startword, n=n, model=model, verbose=verbose)

            while len(sentence) < 25:
                for w in next:
                    sentence.append(w)
                    if w == '</s>':
                        return sentence[:-1]

                next = self._getClosestTo(word=next[-1], n=n, model=model, verbose=verbose)
        else:
            _ngram = self.NGram(n=n, model=model, verbose=verbose)
            highestk = ''

            while len(sentence) < 25 and highestk != '</s>':
                probsforword = {}
                highestv = 0
                highestk = ''
                for k in _ngram['count']:
                    probsforword[k] = self.GetProbability(input=k, n=n, model=model, verbose=verbose)

                    skipchance = random.randint(0, 9)

                    if skipchance == 5:
                        continue

                    if probsforword[k] > highestv and k[0] != '<s>':
                        highestk = k
                        highestv = probsforword[k]

                sentence.append(highestk[0])
                if highestk[0] == '</s>':
                    return sentence[:-1]

        return sentence


class Model(object):
    def __init__(self, corpus, probabilities=None, n=2, model='vanilla', verbose=False):
        if probabilities is None and corpus is None:
            raise Exception('Either a corpus or probabilities must be given.')

        if probabilities is not None:
            self.N = len([w for s in corpus for w in s])
            _probabilities = {}
            for p in probabilities:
                _probabilities[p] = probabilities[p]
            self.probabilities = _probabilities
        else:
            V = 0
            cmodel = model
            if model == 'laplace':
                cmodel = 'vanilla'
                V = len(corpus.NGram(n=1, verbose=verbose)['count'])

            counts = corpus.NGram(n, model=cmodel, verbose=verbose)['count']

            _probabilities = {}
            self.N = len([w for s in corpus for w in s])

            if n is not 1:
                previous = corpus.NGram(n - 1, model=cmodel, verbose=verbose)['count']
                for x in counts:
                    _probabilities[x] = (corpus.GetCount(sequence=x, model=model, verbose=verbose)) / \
                                        (previous[x[:n - 1]] + V)
            else:
                for x in counts:
                    _probabilities[x] = (corpus.GetCount(sequence=x, model=model, verbose=verbose)) / \
                                        (self.N + V)

            self.probabilities = _probabilities

        self.corpus = corpus
        self.model = model

    # ('z', tuple(x, y))
    def GetProbabilityMath(self, forX, givenY: tuple):
        sequence = givenY + (forX,)

        if sequence in self.probabilities:
            return self.probabilities[sequence]
        else:
            if self.model == 'laplace':
                return 1 / self.corpus.GetCount(sequence=givenY, model=self.model)
            else:
                return 0

    def Perplexity(self):
        counts = self.corpus.NGram(n, model=cmodel, verbose=verbose)['count']
        prob = 1
        for p in self.probabilities:
            prob *= self.probabilities[p]
        if prob == 0:
            return float("inf")
        else:
            return prob ** -(1 / self.N)
