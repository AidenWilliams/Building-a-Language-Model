from typing import Union, List
from tqdm.notebook import tqdm
import numpy as np
import random
from LanguageModel.NGramModel import NGramModel
from LanguageModel.NGramCounts import NGramCounts
from LanguageModel.Corpus import Corpus


class LanguageModel(object):
    def __init__(self, corpus=Union[str, List[List[str]], Corpus], ngram=Union[NGramCounts, None],
                 model=Union[NGramModel, None],
                 verbose=False):
        # Get corpus
        if isinstance(corpus, str):
            self.corpus = Corpus.CorpusAsListOfSentences(corpus, verbose)
        else:
            self.corpus = corpus

        # Initialise dictionaries
        self._ngrams = {}
        self._models = {}

        identifier = tuple([1, 'vanilla'])
        if isinstance(ngram, NGramCounts):
            # make sure we have vanilla unigram counts in the model
            if ngram.identifier is not identifier:
                self._ngrams[identifier] = NGramCounts(corpus=self.corpus, n=1, model='vanilla', verbose=verbose)
            self._ngrams[ngram.identifier] = ngram
        else:
            self._ngrams[identifier] = NGramCounts(corpus=self.corpus, n=1, model='vanilla', verbose=verbose)

        if isinstance(model, NGramModel):
            # make sure we have vanilla unigram probabilities in the model
            if model.identifier is not identifier:
                self._models[identifier] = NGramModel(self, n=1, model='vanilla', verbose=verbose)
            self._models[model.identifier] = model
        else:
            self._models[identifier] = NGramModel(self, n=1, model='vanilla', verbose=verbose)

    def GetNGramCounts(self, n=2, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if n < 1:
            raise ValueError('Unigrams and up are supported, otherwise no.')

        identifier = tuple([n, model])
        if identifier in self._ngrams:
            return self._ngrams[identifier]

        self._ngrams[identifier] = NGramCounts(corpus=self.corpus, n=n, model=model, verbose=verbose)
        return self._ngrams[identifier]

    def GetCount(self, sequence: tuple, model):
        _ngram = self.GetNGramCounts(n=len(sequence), model=model)
        return _ngram.GetCount(sequence)

    def GetNGramModel(self, n=2, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if n < 1:
            raise ValueError('Unigrams and up are supported, otherwise no.')

        identifier = tuple([n, model])
        if identifier in self._models:
            return self._models[identifier]

        self._models[identifier] = NGramModel(self, n=n, model=model, verbose=verbose)
        return self._models[identifier]

    # ('z', tuple(x, y))
    def GetProbabilityMath(self, forX, givenY: tuple, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        sequence = givenY + (forX,)
        n = len(sequence)

        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)

        if sequence in _model:
            return _model[sequence]
        else:
            if model == 'laplace':
                return 1 / \
                       self.GetCount(sequence=givenY, model=model)
            else:
                return 0

    def GetProbability(self, input=Union[str, List[str], List[List[str]]],
                       n=2, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if isinstance(input, str):
            tlm = LanguageModel(corpus=[[input]], verbose=verbose)
        else:
            if isinstance(input[0], str):
                tlm = LanguageModel(corpus=[input], verbose=verbose)
            else:
                tlm = LanguageModel(corpus=input, verbose=verbose)

        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)
        _ngram = tlm.GetNGramCounts(n=n, model=model, verbose=verbose)

        input_probability = 1
        exists = False
        for _n in _ngram:
            exists = True
            input_probability *= self.GetProbabilityMath(_n[-1], _n[:n - 1], model=model) ** tlm.GetCount(_n, model=model)

        if not exists:
            input_probability = 0

        return input_probability

    def LinearInterpolation(self, trigram, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if len(trigram) != 3 or type(trigram) != tuple:
            raise Exception('trigram input must be a tuple of 3 words.')

        l1 = 0.1
        l2 = 0.3
        l3 = 0.6

        return l3 * self.GetProbability(input=[trigram[2], trigram[0], trigram[1]], n=3, model=model, verbose=verbose) + \
               l2 * self.GetProbability(input=[trigram[2], trigram[1]], n=2, model=model, verbose=verbose) + \
               l1 * self.GetProbability(input=trigram[2], n=1, model=model, verbose=verbose)

    def Perplexity(self, n=2, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)

        prob = 1
        for p in tqdm(_model, desc='Calculating Perplexity', disable=not verbose):
            prob *= self.GetProbability(p, n=2, model='vanilla', verbose=False)
        if prob == 0:
            return float("inf")
        else:
            return prob ** -(1 / _model.N)

    def _getClosestTo(self, word, n=2, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if n == 1:
            raise ValueError('unigrams are unsupported by this function.')

        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)

        word = word if word == '<s>' else Corpus.filterFurther(word)

        values = [_model[x] for x in _model if x[0] == word]

        weights = np.array(values)
        normalized_weights = weights / np.sum(weights)
        resample_counts = np.random.multinomial(1, normalized_weights)
        chosenKey = list(resample_counts).index(1)
        chosenVal = list(_model)[chosenKey]
        return chosenVal

    def GenerateSentence(self, startword='<s>', n=2, model='vanilla', verbose=False):
        sentence = []
        if startword != '<s>':
            sentence.append(startword)

        if n != 1:
            _model = self.GetNGramModel(n=n, model=model, verbose=verbose)
            next = self._getClosestTo(word=startword, n=n, model=model, verbose=verbose)

            while len(sentence) < 25:
                for w in next:
                    sentence.append(w)
                    if w == '</s>':
                        return sentence[:-1]

                next = self._getClosestTo(word=next[-1], n=n, model=model, verbose=verbose)
        else:
            unigram = self.GetNGramModel(n=1, model=model, verbose=verbose)
            chosenVal = ''

            while len(sentence) < 25 and chosenVal != '</s>':
                weights = np.array(list(unigram.values()))
                normalized_weights = weights / np.sum(weights)
                resample_counts = np.random.multinomial(1, normalized_weights)
                chosenKey = list(resample_counts).index(1)
                chosenVal = list(unigram)[chosenKey]

                sentence.append(chosenVal[0])
                if chosenVal == '</s>':
                    return sentence[:-1]

        return sentence
