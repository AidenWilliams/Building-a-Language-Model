from typing import Union

from LanguageModel.NGramModel import NGramModel
from LanguageModel.NGramCounts import NGramCounts
from LanguageModel.Corpus import Corpus


class LanguageModel(object):
    def __init__(self, corpus=Union[str, Corpus], ngram=Union[NGramCounts, None], model=Union[NGramModel, None],
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
        if ngram is not None:
            # make sure we have vanilla unigram counts in the model
            if ngram.identifier is not identifier:
                self._ngrams[identifier] = NGramCounts(corpus=self.corpus, n=1, model='vanilla', verbose=verbose)
            self._ngrams[ngram.identifier] = ngram
        else:
            self._ngrams[identifier] = NGramCounts(corpus=self.corpus, n=1, model='vanilla', verbose=verbose)

        if model is not None:
            # make sure we have vanilla unigram probabilities in the model
            if model.identifier is not identifier:
                self._models[identifier] = NGramModel(corpus=self.corpus, n=1, model='vanilla', verbose=verbose)
            self._models[model.identifier] = model
        else:
            self._models[identifier] = NGramModel(corpus=self.corpus, n=1, model='vanilla', verbose=verbose)

    def GetNGramCounts(self, n=2, model='vanilla', verbose=False):
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        identifier = tuple([n, model])
        if identifier in self._ngrams:
            return self._ngrams[identifier]

        self._ngrams[identifier] = NGramCounts(corpus=self.corpus, n=n, model=model, verbose=verbose)
        return self._ngrams[identifier]

    def GetNGramModel(self, n=2, model='vanilla', verbose=False):
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        identifier = tuple([n, model])
        if identifier in self._models:
            return self._models[identifier]

        self._models[identifier] = NGramModel(corpus=self.corpus, n=n, model=model, verbose=verbose)
        return self._models[identifier]
