from LanguageModel.Corpus import Corpus
from tqdm.notebook import tqdm
from collections import defaultdict


class NGramCounts(object):
    def __init__(self, corpus: Corpus, n=2, model='vanilla', verbose=False):
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        if model == 'laplace' or model == 'vanilla':
            counts = self.Counts(n=n, corpus=corpus, verbose=verbose)

        else:
            _count = self.Counts(n=1, corpus=corpus, verbose=verbose)
            tc = []
            for s in corpus:
                ts = []
                for w in s:
                    if _count[tuple([w])] < 3:
                        ts.append('UNK')
                    else:
                        ts.append(w)
                tc.append(ts)

            counts = self.Counts(n=n, corpus=tc, verbose=verbose)

        self.identifier = tuple([n, model])
        self._ngram = counts

    def __repr__(self):
        return self._ngram

    def __getitem__(self, item):
        return self._ngram[item]

    @staticmethod
    def Counts(n, corpus: Corpus, verbose=False):
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
        if sequence in self:
            return self[sequence] + int(self.identifier[1] == 'laplace')
        else:
            if self.identifier[1] == 'laplace':
                return 1
            else:
                return 0
