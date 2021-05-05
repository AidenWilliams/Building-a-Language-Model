from LanguageModel.NGramCounts import NGramCounts
from LanguageModel import LanguageModel


class NGramModel(object):
    def __init__(self, lm: LanguageModel, testProbabilities=None, n=2, model='vanilla', verbose=False):
        self.identifier = tuple([n, model])
        self.lm = lm

        if testProbabilities is not None:
            _probabilities = {}
            for p in testProbabilities:
                _probabilities[p] = testProbabilities[p]
            self._probabilities = _probabilities
        else:
            V = 0
            cmodel = model
            if model == 'laplace':
                cmodel = 'vanilla'
                V = len(NGramCounts(corpus=lm.corpus, n=1, verbose=verbose)['count'])

            ngram = lm.GetNGramCounts(n=n, model=cmodel, verbose=verbose)

            _probabilities = {}
            self.N = len([w for s in lm.corpus for w in s])

            if n is not 1:
                previous = lm.GetNGramCounts(n=n - 1, model=cmodel, verbose=verbose)['count']
                for x in ngram:
                    _probabilities[x] = (ngram.GetCount(sequence=x, model=model, verbose=verbose)) / \
                                        (previous[x[:n - 1]] + V)
            else:
                for x in ngram:
                    _probabilities[x] = (ngram.GetCount(sequence=x, model=model, verbose=verbose)) / \
                                        (self.N + V)

            self._probabilities = _probabilities

    def __repr__(self):
        return self._probabilities

    def __getitem__(self, item):
        return self._probabilities[item]

    # move to Language Model
    # ('z', tuple(x, y))
    def GetProbabilityMath(self, forX, givenY: tuple):
        sequence = givenY + (forX,)

        if sequence in self:
            return self[sequence]
        else:
            if self.identifier[1] == 'laplace':
                return 1 / \
                       self.lm.GetNGramCounts(n=self.identifier[0], model=self.identifier[1]).GetCount(sequence=givenY)
            else:
                return 0
