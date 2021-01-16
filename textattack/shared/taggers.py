from abc import abstractmethod

import flair
import nltk

from textattack.shared import utils

flair.device = utils.device


class PosTagger(metaclass=utils.Singleton):
    """Base class for part-of-speech (POS) tagging."""

    @abstractmethod
    def __call__(self, words):
        """Tag a list of words with their part-of-speech.

        Args:
            words (list[str]): List of words to tag.
            univeral_tagset (bool): Use the Universal Dependency tag set. If false, use Penn Treebank POS tag set.
        Returns:
            list[str]: List of corresponding POS tags.
        """


class NerTagger(metaclass=utils.Singleton):
    """Base class for NER tagging."""

    @abstractmethod
    def __call__(self, words):
        """Accepts a list of words and returns a list of NER tags.

        If word does not have a corresponding NER tag, simply assign
        None.
        """
        raise NotImplementedError()


class EnglishPosTagger(PosTagger):
    """Part-of-speech Tagger for English text. Three options are available for
    POS tagging.

        - flair: Use the tagger from Flair library (https://github.com/flairNLP/flair)
        - stanza: Use the tagger from Stanza library (https://github.com/stanfordnlp/stanza)
        - nltk: Use the tagger from NLTK library (https://www.nltk.org)

    Args:
        tagger (str): Name of tagger to use. Available options are "flair", "stanza", "nltk". Default is "flair".
    """

    def __init__(self, tagger="flair"):
        if tagger == "flair":
            # We load the universal tagset tagger first because it's more likely to be used.
            # If we need to use Penn Treebank POS tagset, load the tagger lazily.
            self.upos_tagger = flair.models.SequenceTagger.load("upos-fast")
            self.pos_tagger = None
        elif tagger == "stanza":
            stanza = utils.LazyLoader("stanza", globals(), "stanza")
            self.pos_tagger = stanza.Pipeline(
                lang="en", processors="tokenize, pos", tokenize_pretokenized=True
            )
        elif tagger == "nltk":
            # NLTK doesn't require loading a tagger.
            pass
        else:
            raise ValueError(
                f'Available options for `tagger` are {["flair", "stanza", "nltk"]}.'
            )

        self._tagger_name = tagger

    def _run_stanza(self, words, universal_tagset):
        text = " ".join(words)
        doc = self.pos_tagger(text)
        pos_list = []
        for sent in doc.sentences:
            for word in sent.words:
                if universal_tagset:
                    pos_list.append(word.upos)
                else:
                    pos_list.append(word.xpos)
        if len(words) != len(pos_list):
            raise ValueError("Length of word list and POS tag list is different")

        return pos_list

    def _run_flair(self, words, universal_tagset):
        text = " ".join(words)
        sent = flair.data.Sentence(text, use_tokenizer=lambda x: x.split())
        if universal_tagset:
            tagger = self.upos_tagger
        else:
            self.pos_tagger = flair.models.SequenceTagger("pos-fast")
            tagger = self.pos_tagger
        tagger.predict(sent)
        pos_list = [token.annotation_layers["pos"][0]._value for token in sent.tokens]
        if len(words) != len(pos_list):
            raise ValueError("Length of word list and POS tag list is different")

        return pos_list

    def __call__(self, words, universal_tagset=True):
        """Tag a list of words with their part-of-speech.

        Args:
            words (list[str]): List of words to tag.
            univeral_tagset (bool): Use the Universal Dependency tag set. If false, use Penn Treebank POS tag set.
        Returns:
            list[str]: List of corresponding POS tags.
        """
        if self._tagger_name == "flair":
            return self._run_flair(words, universal_tagset)
        elif self._tagger_name == "stanza":
            return self._run_stanza(words, universal_tagset)
        else:
            if universal_tagset:
                return zip(*nltk.pos_tag(words, tagset="universal"))[1]
            else:
                return zip(*nltk.pos_tag(words))[1]


class EnglishNerTagger(NerTagger):
    """Base class for NER tagging."""

    @abstractmethod
    def __call__(self, words):
        """Accepts a list of words and returns a list of NER tags.

        If word does not have a corresponding NER tag, simply assign
        None.
        """
        raise NotImplementedError()


class KoreanNerTagger(NerTagger):
    """Part-of-speech Tagger for Korea text.

    Uses `khaiii` library.
    """

    @abstractmethod
    def __call__(self, words):
        """Accepts a list of words and returns a list of NER tags.

        If word does not have a corresponding NER tag, simply assign
        None.
        """
        raise NotImplementedError()
