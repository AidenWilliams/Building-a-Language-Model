from LanguageModel.NGramCounts import NGramCounts
import LanguageModel
from tqdm.notebook import tqdm


class NGramModel(object):
    """The NGramModel class represent a N-Gram probability model of the Language Model.

    The object can be identified for any n > 0 and any of the 3 models: "vanilla"/"laplace"/"unk".

    Attributes
    ----------
    identifier :  tuple([int, str])
        a tuple of the n and model of the NGramCounts.
    N : int
        a int number of the number of words in the provided LanguageModel corpus.
    _probabilities :  {tuple([str]*n): probability}
        a dict containing n sized sequences in the form of a tuple with their percentage share in the provided
        LanguageModel corpus.
    """
    def __init__(self, lm: LanguageModel, testProbabilities=None, n=2, model='vanilla', verbose=False):
        """Initialises the NGramModel from corpus given the identifier combo and a LanguageModel. The NGramModel can
        also be 'copy' constructed from a dict {tuple([str]*n): probability}.

        The process of probability calculation is done in detail via in line comments.

        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"
            or
            If n is smaller than 1
        """
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
            # Define V as 0 first
            V = 0
            # Define if model is laplace or unk, define it as the length of the respective unigram count
            if model == 'laplace':
                V = len(NGramCounts(corpus=lm.corpus, model=model, n=1, verbose=verbose))
            elif model == 'unk':
                V = len(NGramCounts(corpus=lm.corpus, model=model, n=1, verbose=verbose))

            # Get the ngram that will be used
            ngram = lm.GetNGramCounts(n=n, model=model, verbose=verbose)

            self._probabilities = {}

            # Two methods are used, based on whether n is 1 or not
            # In any case V is added to the denominator
            if n is not 1:
                # Get the previous NGram count, previous meaning n-1, so if we are getting calculating the bigram
                # probabilities we get the unigram counts.
                previous = lm.GetNGramCounts(n=n - 1, model=model, verbose=verbose)
                # Now for every sequence in ngram we calculate its probability.
                for x in tqdm(ngram, desc='Calculating Probabilities', disable=not verbose):
                    # Calculation is done by dividing the count of the ngram by the count of its history.
                    # Here the history is defined as x[:n-1]. If x is a tuple(['x', 'y', 'z']) then n must be 3,
                    # x[:n - 1] gives us ('x', 'y'). This history count is taken from the previous ngram count.
                    self._probabilities[x] = (ngram.GetCount(sequence=x)) / \
                                             (previous.GetCount(x[:n - 1]) + V)
            else:
                # Unigram counting is simpler.
                # The count of each word is divided by N.
                for x in tqdm(ngram, desc='Calculating Probabilities', disable=not verbose):
                    self._probabilities[x] = (ngram.GetCount(sequence=x)) / \
                                             (self.N + V)

    def __repr__(self):
        """Allows the representation of NGramModel as the _probabilities dict

        Returns
        -------
        _probabilities
        """
        return self._probabilities

    def __iter__(self):
        """Gives functionality to iterate over _probabilities
        """
        for sequence in self._probabilities:
            yield sequence

    def __getitem__(self, item):
        """
        Returns
        -------
        item in _probabilities at index
        """
        return self._probabilities[item]

    def values(self):
        """
        Returns
        -------
        _probabilities' values
        """
        return self._probabilities.values()
