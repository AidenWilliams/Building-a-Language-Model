from typing import Union, List
from tqdm.notebook import tqdm
from mpmath import mp
import random
from LanguageModel.NGramModel import NGramModel
from LanguageModel.NGramCounts import NGramCounts
from LanguageModel.Corpus import Corpus


# TODO: parameter documentation

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
        _ngram = self.GetNGramCounts(n=n, model=model, verbose=verbose)

        if sequence in _model:
            return _model[sequence]
        else:
            if model != 'vanilla':
                return 1 / \
                       _ngram.GetCount(sequence=givenY)
            else:
                return 0

    def GetProbability(self, input=Union[str, List[str], List[List[str]]],
                       n=2, model='vanilla'):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if isinstance(input, str):
            tlm = LanguageModel(corpus=[[input]])
        else:
            if isinstance(input[0], str):
                tlm = LanguageModel(corpus=[input])
            else:
                tlm = LanguageModel(corpus=input)

        _model = self.GetNGramModel(n=n, model=model)
        _ngram = tlm.GetNGramCounts(n=n, model=model)

        input_probability = 1
        exists = False
        for _n in _ngram:
            exists = True
            input_probability *= self.GetProbabilityMath(_n[-1], _n[:n - 1], model=model) ** _ngram.GetCount(_n)

        if not exists:
            input_probability = 0

        return input_probability

    def LinearInterpolation(self, trigram, model='vanilla'):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if len(trigram) != 3 or type(trigram) != tuple:
            raise Exception('trigram input must be a tuple of 3 words.')

        l1 = 0.1
        l2 = 0.3
        l3 = 0.6

        return l3 * self.GetProbability(input=[trigram[2], trigram[0], trigram[1]], n=3, model=model) + \
               l2 * self.GetProbability(input=[trigram[2], trigram[1]], n=2, model=model) + \
               l1 * self.GetProbability(input=trigram[2], n=1, model=model)

    def Perplexity(self, n=2, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)

        prob = mp.mpf(1)
        for p in tqdm(_model, desc='Calculating Perplexity', disable=not verbose):
            prob *= self.GetProbability(p, n=n, model='vanilla')
        if prob == 0:
            return mp.mpf("inf")
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

        probabilities = [_model[x] for x in _model if x[0] == word]
        keys = [x for x in _model if x[0] == word]

        if probabilities is not None:
            return random.choices(list(keys), weights=probabilities, k=1)[0][1:]
        else:
            return '</s>'

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
            _model = self.GetNGramModel(n=1, model=model, verbose=verbose)
            word = '<s>'

            while len(sentence) < 25 and word != '</s>':
                probabilities = [_model[x] for x in _model]
                word = random.choices(list(_model), weights=probabilities, k=1)[0][0]

                if word != '<s>':
                    sentence.append(word)
                if word == '</s>':
                    return sentence[:-1]

        return sentence
