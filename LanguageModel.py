from lxml import etree
import re
from tqdm.notebook import tqdm
import os


class Corpus(object):
    def __init__(self, corpus=None, directory='Corpus/'):
        if corpus is None:
            self._corpus = self._CorpusAsListOfSentences(directory)
        else:
            self._corpus = corpus
        identifier = tuple([1, 'vanilla'])
        self._ngrams = {}
        self._ngrams[identifier] = self.NGram(n=1)
        self._models = {}
        self._models[identifier] = self.Model(n=1)

    def __len__(self):
        return len(self._corpus)

    def __iter__(self):
        for word in self._corpus:
            yield word

    def __getitem__(self, index):
        return self._corpus[index]

    # Create functions to free memory once function scope is left
    @staticmethod
    def _ReadCorpus(root='Corpus/'):
        if not os.access(root, os.R_OK):
            print('Check root!!')

        xml_data = []

        for file in tqdm(os.listdir(root), desc='Reading Files'):
            xml_data.append(open(os.path.join(root, file), 'r', encoding='utf8').read())  # Read file
        return xml_data

    @staticmethod
    def _ParseAsXML(root='Corpus/'):
        parser = etree.XMLParser(recover=True)
        roots = []
        xml_data = Corpus._ReadCorpus(root)
        for xml in tqdm(xml_data, desc='Parsing XML'):
            roots.append(etree.fromstring(xml, parser=parser))
        return roots

    @staticmethod
    def _CorpusAsListOfSentences(root='Corpus/'):
        roots = Corpus._ParseAsXML(root)
        sentences = []
        for root in tqdm(roots, desc='XML File'):
            for i, p in tqdm(enumerate(root), desc='Paragraph'):
                for k, s in enumerate(p):
                    unfiltered_sentence = re.split(r'\n', s.text.lstrip('\n'))
                    sentence = []
                    for unfiltered_word in unfiltered_sentence:
                        if unfiltered_word is not '':
                            filtered_word = unfiltered_word.split('\t')
                            sentence.append(filtered_word[0])

                    if sentence is not []:
                        sentence.insert(0, '<s>')
                        sentences.append(sentence)
                        sentence.append('</s>')
        return sentences

    def Counts(self, n):
        counts = {}
        for s in tqdm(self, desc='Counting x counts'):
            for i in range(len(s) + 1):
                if i < n:
                    continue
                count = []
                for x in range(n, 0, -1):
                    count.append(s[i - x])
                count = tuple(count)

                if count in counts:
                    counts[count] += 1
                else:
                    counts[count] = 1

        return counts

    def NGram(self, n=2, model='vanilla'):
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
            counts = self.Counts(n=n)

            if model == 'laplace':
                for x in counts:
                    counts[x] += 1

        elif model == 'unk':
            _count = self.Counts(n=1)
            tc = []
            for s in self:
                ts = []
                for w in s:
                    if _count[tuple([w])] < 3:
                        ts.append('UNK')
                    else:
                        ts.append(w)
                tc.append(ts)

            temp = Corpus(corpus=tc)
            counts = temp.Counts(n=n)

        result = {
            'count': counts,
            'model': model
        }

        self._ngrams[identifier] = result
        return self._ngrams[identifier]

    def GetCount(self, sequence: tuple, model='vanilla'):
        n = len(sequence)
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        _ngram = self.NGram(n, model)['count']

        if sequence in _ngram:
            return _ngram[sequence]
        else:
            if model == 'laplace':
                return 1
            else:
                return 0

    def Model(self, n=2, model='vanilla'):
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

        self._models[identifier] = Model(corpus=self, n=n, model=model)
        return self._models[identifier]

    def GetProbability(self, input, n, model='vanilla'):
        if n < 1:
            raise Exception('Unigrams and up are supported, otherwise no.')

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        if type(input) is str:
            tc = Corpus([[input]])
        else:
            paragraph = False
            for elem in input:
                if isinstance(elem, list):
                    paragraph = True
                if paragraph and not isinstance(elem, list):
                    raise Exception('Input must be of the forms:\nstr\n[str]\n[[str],[str]].')

            if paragraph:
                tc = Corpus(input)
            else:
                tc = Corpus([input])

        _model = self.Model(n=n, model=model)
        _ngram = tc.NGram(n=n, model=model)

        input_probability = 1
        exists = False
        for _n in _ngram['count']:
            exists = True
            input_probability *= _model.GetProbabilityMath(_n[-1], _n[:n - 1]) ** tc.GetCount(_n, model=model)

        if not exists:
            input_probability = 0

        return input_probability

    def TrigramLinearInterpolation(self, input, model='vanilla'):
        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise Exception('Only "vanilla"/"laplace"/"unk" models are supported.')

        paragraph = False
        for elem in input:
            if isinstance(elem, list):
                paragraph = True
            if paragraph and not isinstance(elem, list):
                raise Exception('Input must be of the forms:\n[str, str, str]\n[[str, str, str], ...,  [str, str, '
                                'str]].')

        if paragraph:
            tc = Corpus(input)
        else:
            tc = Corpus([input])

        l1 = 0.1
        l2 = 0.3
        l3 = 0.6

        _ngram = tc.NGram(n=3, model=model)

        input_probability = 1
        exists = False
        for _n in _ngram['count']:
            exists = True
            input_probability *= l3 * self.GetProbability(input=[_n[2], _n[:2][0], _n[:2][1]], n=3, model=model) + \
                                 l2 * self.GetProbability(input=[_n[2], _n[1]], n=2, model=model) + \
                                 l1 * self.GetProbability(input=_n[2], n=1, model=model)

        if not exists:
            input_probability = 0

        return input_probability


class Model(object):
    def __init__(self, corpus, n=2, model='vanilla'):
        V = 0
        cmodel = model
        if model == 'laplace':
            cmodel = 'vanilla'
            V = len(corpus.NGram(n=1)['count'])

        counts = corpus.NGram(n, model=cmodel)['count']

        probabilities = {}
        self.N = len([w for s in corpus for w in s])

        if n is not 1:
            previous = corpus.NGram(n - 1, model=cmodel)['count']
            for x in counts:
                probabilities[x] = {
                    'probability': (counts[x] + int(model == 'laplace')) / (previous[x[:n - 1]] + V)}
        else:
            for x in counts:
                probabilities[x] = {'probability': (counts[x] + int(model == 'laplace')) / (self.N + V)}
        self.probabilities = probabilities
        self.model = model

    # ('z', tuple(x, y))
    def GetProbabilityMath(self, forX, givenY):
        sequence = givenY + (forX,)

        if sequence in self.probabilities:
            return self.probabilities[sequence]['probability']
        else:
            return 0

    def Perplexity(self):
        prob = 1
        for p in self.probabilities:
            prob *= self.probabilities[p]['probability']

        return prob ** -(1 / self.N)


# corpus = Corpus(directory='Test Corpus/')
# #
# t = ['I', 'am', 'Sam']
#
# print(corpus.LinearInterpolation(t))
# #
# # # corpus.NGram()
# # # corpus.Model()
