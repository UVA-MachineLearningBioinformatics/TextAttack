from abc import ABC, abstractmethod

from textattack.shared import utils


class LanguageResource(ABC, metaclass=utils.Singleton):

    """Singleton class responsible for provisioning resources for different
    languages. This is not meant to override any of user's choices, but just
    provide default resources. Provisioned resources include:

    - `word_segmenter`: A single callable that accepts a string and returns a list of words of the string.
    - `masked_lm_name`: Name of pretrained Transformers model to use for masked language modeling.
    - `word_embedding`: Default word embedding to use for the language. It should be of class `textattack.shared.AbstractWordEmbedding`.
    - `word_net_lang_code`: Code of the language for
    """

    @abstractmethod
    @property
    def word_segmenter(self):
        """Returns a callable that accepts a string as its only argument and
        returns list of words."""
        raise NotImplementedError()

    @abstractmethod
    @property
    def word_embedding(self):
        """Returns the."""
        raise NotImplementedError()

    @abstractmethod
    @property
    def pos_tagger(self):
        raise NotImplementedError()

    @abstractmethod
    @property
    def masked_lm_name(self):
        """Returns the name of pretrained transformers model to use for masked
        langauge modeling."""
        raise NotImplementedError()

    @abstractmethod
    @property
    def wordnet_lang_code(self):
        raise NotImplementedError()
