from LanguageModel.Corpus import Corpus
from tqdm.notebook import tqdm
from collections import defaultdict
from typing import Union, List


class NGramCounts(object):
    """The NGramCounts class represents a N-Gram count of the Language Model.

    The object can be identified for any n > 0 and any of the 3 models: "vanilla"/"laplace"/"unk".

    Attributes
    ----------
    identifier :  tuple([int, str])
        a tuple of the n and model of the NGramCounts.

    _ngram :  defaultdict = {tuple([str]*n): count}
        a dict containing n sized sequences in the form of a tuple with their count in corpus.
    """
    def __init__(self, corpus=Union[List[List[str]], Corpus], n=2, model='vanilla', verbose=False):
        """Initialises the NGramCounts from corpus given the identifier combo

        If the model is specified to be laplace or vanilla the counts of n sized sequences are taken from the
        available corpus. Otherwise, if the model is unk, a temp corpus is created using the vanilla unigram counts.
        If a word is written less than 3 times it is omitted from the new corpus. After this step the counts of n
        sized sequences are taken from the temp corpus.

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

        if n < 1:
            raise ValueError('Unigrams and up are supported, otherwise no.')

        if model == 'laplace' or model == 'vanilla':
            counts = self.Counts(n=n, corpus=corpus, verbose=verbose)

        else:
            _count = self.Counts(n=1, corpus=corpus, verbose=verbose)
            tc = []
            for s in corpus:
                ts = []
                for w in s:
                    if _count[tuple([w])] < 3:
                        ts.append('unk')
                    else:
                        ts.append(w)
                tc.append(ts)

            counts = self.Counts(n=n, corpus=tc, verbose=verbose)

        self.identifier = tuple([n, model])
        self._ngram = counts

    def __repr__(self):
        """Allows the representation of NGramCounts as the _ngram dict

        Returns
        -------
        _ngram
        """
        return self._ngram

    def __iter__(self):
        """Gives functionality to iterate over _ngram
        """
        for sequence in self._ngram:
            yield sequence

    def __getitem__(self, item):
        """
        Returns
        -------
        item in _ngram at index
        """
        return self._ngram[item]

    def __len__(self):
        """
        Returns
        -------
        length of _ngram
        """
        return len(self._ngram)

    @staticmethod
    def Counts(n, corpus=Union[List[List[str]], Corpus], verbose=False):
        """ Counts the n sized sequences in the given corpus.

        This is done by looping over each sentence, gathering a tuple of each n sized sequence and counting its
        occurrences.

        Returns
        -------
        counts in a dictionary of this form:

        {tuple([str]*n): count}

        Where sequence is a tuple of size n and count is an integer containing the number of counts.
        """
        counts = defaultdict(lambda: 0)

        for s in tqdm(corpus, desc='Counting x counts', disable=not verbose):
            for i in range(len(s) + 1):
                if i < n:
                    continue
                sequence = []
                for x in range(n, 0, -1):
                    sequence.append(s[i - x])
                sequence = tuple(sequence)

                counts[sequence] += 1

        return counts

    def GetCount(self, sequence: tuple):
        """Returns the count of a given n sized sequence.

        Laplace smoothing's count addition is done at this stage.

        Returns
        -------
        The count for sequence according to the class' model.
        """
        if sequence in self:
            return self[sequence] + int(self.identifier[1] != 'vanilla')
        else:
            return int(self.identifier[1] != 'vanilla')
