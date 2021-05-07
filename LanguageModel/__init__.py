from typing import Union, List
from tqdm.notebook import tqdm
from mpmath import mp
import random
from LanguageModel.NGramModel import NGramModel
from LanguageModel.NGramCounts import NGramCounts
from LanguageModel.Corpus import Corpus


# TODO: parameter documentation

class LanguageModel(object):
    """The LanguageModel class represents a complete (as far as the assignment requires) N Gram Language Model.

    The class is intended to be used for NGram frequency and model generation as well as getting probability for given
    inputs, model perplexity calculation and sentence generation.

    Attributes
    ----------
    corpus :  Corpus
        the corpus for the LanguageModel.

    _ngrams : dict
        a dictionary of NGramCounts.

    _models :  dict
        a dictionary of NGramModels.
    """
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
            if ngram.identifier is not identifier:
                # make sure we have vanilla unigram counts in the LanguageModel
                self._ngrams[identifier] = NGramCounts(corpus=self.corpus, n=1, model='vanilla', verbose=verbose)
            # add passed ngram to _ngrams
            self._ngrams[ngram.identifier] = ngram
        else:
            # make sure we have vanilla unigram counts in the LanguageModel
            self._ngrams[identifier] = NGramCounts(corpus=self.corpus, n=1, model='vanilla', verbose=verbose)

        if isinstance(model, NGramModel):
            if model.identifier is not identifier:
                # make sure we have vanilla unigram probabilities in the LanguageModel
                self._models[identifier] = NGramModel(self, n=1, model='vanilla', verbose=verbose)
            # add passed model to _models
            self._models[model.identifier] = model
        else:
            # make sure we have vanilla unigram probabilities in the LanguageModel
            self._models[identifier] = NGramModel(self, n=1, model='vanilla', verbose=verbose)

    def GetNGramCounts(self, n=2, model='vanilla', verbose=False):
        """Gets or generates an NGramCounts object depending if an n and model identified _ngram is found _ngrams.
        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"
            or
            If n is smaller than 0

        Returns
        -------
        NGramCounts for n and model.
        """
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
        """Gets or generates an NGramModel object depending if an n and model identified model is found _models.
        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"
            or
            If n is smaller than 0

        Returns
        -------
        NGramModel for n and model.
        """
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

    def GetProbabilityMath(self, forX, givenY: tuple, model='vanilla', verbose=False):
        """Gets the probability forX givenY, for the given model.

        The function is labeled 'Math' because the input is expected to be in the standard probability way of : P(A | B)
        which in python makes forX = A, givenY=B and in practice, for an ngram (x,y,z), forX = z, givenY = tuple(x, y)).

        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"

        Returns
        -------
        probability forX givenY
        """
        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        sequence = givenY + (forX,)
        n = len(sequence)

        # Get the necessary model and counts
        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)
        _ngram = self.GetNGramCounts(n=n, model=model, verbose=verbose)

        if sequence in _model:
            # Return the probability if found
            return _model[sequence]
        else:
            # Calculate the laplace smoothed probability if the model isn't vanilla
            if model != 'vanilla':
                return 1 / \
                       _ngram.GetCount(sequence=givenY)
            # or return 0
            else:
                return 0

    def GetProbability(self, input=Union[str, List[str], List[List[str]]], n=2, model='vanilla'):
        """Gets the probability input given n and model.

        The description of probability calculation is done in detail via in line comments.

        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"
            or
            If n is smaller than 0

        Returns
        -------
        probability input appearing in this LanguageModel for n and model
        """

        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if n < 1:
            raise ValueError('Unigrams and up are supported, otherwise no.')

        # Generate the temp LanguageModel depending on the input
        if isinstance(input, str):
            tlm = LanguageModel(corpus=[[input]])
        else:
            if isinstance(input[0], str):
                tlm = LanguageModel(corpus=[input])
            else:
                tlm = LanguageModel(corpus=input)

        # Get the n, model NGramModel from this LanguageModel
        _model = self.GetNGramModel(n=n, model=model)
        # Get the n, model NGramCounts from the temp LanguageModel
        _ngram = tlm.GetNGramCounts(n=n, model=model)

        # initialise the return as 1
        input_probability = mp.mpf(1)
        #exists = False
        # for every ngram in _ngram
        for _n in _ngram:
            #exists = True
            # Raise _n's probability in this LanguageModel to its count.
            # This is done since repeated ngrams in input would be stripped into NGramCounts, and in this loop will not
            # be found again.
            input_probability *= self.GetProbabilityMath(_n[-1], _n[:n - 1], model=model) ** _ngram.GetCount(_n)

        # if not exists:
        #     input_probability = 0

        return input_probability

    def LinearInterpolation(self, trigram, model='vanilla'):
        """ Gets the probability for a trigram using Linear Interpolation, given model.

        The lambdas are defined as: unigram = 0.1, bigram = 0.3, trigram = 0.6.

        Then the Trigram Linear Interpolation is calculated by adding the probabilities of each part the trigram.
        Each probability is also multiplied to its respective lambda.

        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"
        Exception
            If the trigram is not in fact a trigram

        Returns
        -------
        probability for a trigram using Linear Interpolation, given model.
        """
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
        """ Calculates the Perplexity for the NGramModel with identifier n, model.

        Perplexity is calculated by first multiplying all the probabilities in the NGramModel and then raising this
        result to  -(1 / N).

        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"
            or
            If n is smaller than 0

        Returns
        -------
        Perplexity for the NGramModel with identifier n, model.
        """
        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if n < 1:
            raise ValueError('Unigrams and up are supported, otherwise no.')

        # Get the model
        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)

        prob = mp.mpf(1)
        # Multiply all the probabilities in the NGramModel
        for p in tqdm(_model, desc='Calculating Perplexity', disable=not verbose):
            prob *= self.GetProbability(p, n=n, model='vanilla')

        # Shouldn't be 0 because of mp but jic
        if prob == 0:
            return 0
        else:
            # raise result of multiplications to -(1 / N).
            return prob ** -(1 / _model.N)

    def _getClosestTo(self, word, n=2, model='vanilla', verbose=False):
        """ Gets the ngram closest to word for the NGramModel with identifier n, model.

        The ngram is chosen randomly depending on its weight related to the given word.

        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"
            or
            If n is smaller than 0

        Returns
        -------
        ngram closest to word for the NGramModel with identifier n, model.
        """
        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if n == 1:
            raise ValueError('unigrams are unsupported by this function.')

        # Get the model
        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)
        # Filter word
        word = word if word == '<s>' else Corpus.filterFurther(word)

        # Get the weighted probabilities
        probabilities = [_model[x] for x in _model if x[0] == word]
        # Get the keys
        keys = [x for x in _model if x[0] == word]

        if probabilities is not None:
            # If there's something choose 1 ngram given the weights
            # [0] is added as random.choices returns a list
            # and [1:] to remove the word from the tuple chosen
            return random.choices(list(keys), weights=probabilities, k=1)[0][1:]
        else:
            return '</s>'

    def GenerateSentence(self, startword='<s>', n=2, model='vanilla', verbose=False):
        """ Generates a sentence from startword given n and model.

        The description of generation is done in detail via in line comments.

        Raises
        ------
        ValueError
            If the model inputted is not "vanilla"/"laplace"/"unk"
            or
            If n is smaller than 0

        Returns
        -------
        A generated sentence in the form of list[str]
        """
        if model != 'vanilla' and \
                model != 'laplace' and \
                model != 'unk':
            raise ValueError('Only "vanilla"/"laplace"/"unk" models are supported.')

        if n == 1:
            raise ValueError('unigrams are unsupported by this function.')
        # Initialise the return sentence
        sentence = []
        # Add startword if it isnt the start token
        if startword != '<s>':
            sentence.append(startword)

        # Get the model
        _model = self.GetNGramModel(n=n, model=model, verbose=verbose)

        # Two methods are used, based on whether n is 1 or not
        if n != 1:
            # get the first next set of words
            next = self._getClosestTo(word=startword, n=n, model=model, verbose=verbose)

            # Continue generating until either </s> is found or the sentence is 25 words long.
            # 25 is purely arbitrary
            while len(sentence) < 25:
                # Add each word found to sentence
                for w in next:
                    sentence.append(w)
                    if w == '</s>':
                        return sentence[:-1]

                # get the next set of words
                next = self._getClosestTo(word=next[-1], n=n, model=model, verbose=verbose)
        else:
            # similar to the _getClosestTo function...
            # Continue generating until either </s> is found or the sentence is 25 words long.
            while len(sentence) < 25 and startword != '</s>':
                # Get the weighted probabilities
                probabilities = [_model[x] for x in _model]
                # Get a random word from the model
                startword = random.choices(list(_model), weights=probabilities, k=1)[0][0]

                # Ignore start tokens
                if startword != '<s>':
                    sentence.append(startword)
                # Stop when the end token is found
                if startword == '</s>':
                    return sentence[:-1]

        return sentence
