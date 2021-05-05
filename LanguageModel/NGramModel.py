from LanguageModel.NGramCounts import NGramCounts
import LanguageModel


class NGramModel(object):
    def __init__(self, lm: LanguageModel, testProbabilities=None, n=2, model='vanilla', verbose=False):

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')


        self.identifier = tuple([n, model])
        self.N = len([w for s in lm.corpus for w in s])

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
                V = len(NGramCounts(corpus=lm.corpus, n=1, verbose=verbose))

            ngram = lm.GetNGramCounts(n=n, model=cmodel, verbose=verbose)

            _probabilities = {}

            if n is not 1:
                previous = lm.GetNGramCounts(n=n - 1, model=cmodel, verbose=verbose)
                for x in ngram:
                    _probabilities[x] = (ngram.GetCount(sequence=x)) / \
                                        (previous.GetCount(x[:n - 1]) + V)
            else:
                for x in ngram:
                    _probabilities[x] = (ngram.GetCount(sequence=x)) / \
                                        (self.N + V)

            self._probabilities = _probabilities

    def __repr__(self):
        return self._probabilities

    def __iter__(self):
        for sequence in self._probabilities:
            yield sequence

    def __getitem__(self, item):
        return self._probabilities[item]

