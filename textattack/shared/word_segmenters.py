from abc import abstractmethod
import string

from textattack.shared import utils


class WordSegmenter(metaclass=utils.Singleton):
    """Base class for word segmentation."""

    # If part-of-speech of text can be obtained for free during segmentation, set this to be True.
    POS_TAGGING_INCLUDED = False

    @abstractmethod
    def __call__(self, text):
        """Accepts single string and returns a list of words and a list of
        start positions of each of the words."""
        raise NotImplementedError()


class EnglishWordSegmenter(WordSegmenter):

    HOMOGLYPHS = {
            "˗",
            "৭",
            "Ȣ",
            "𝟕",
            "б",
            "Ƽ",
            "Ꮞ",
            "Ʒ",
            "ᒿ",
            "l",
            "O",
            "`",
            "ɑ",
            "Ь",
            "ϲ",
            "ԁ",
            "е",
            "𝚏",
            "ɡ",
            "հ",
            "і",
            "ϳ",
            "𝒌",
            "ⅼ",
            "ｍ",
            "ո",
            "о",
            "р",
            "ԛ",
            "ⲅ",
            "ѕ",
            "𝚝",
            "ս",
            "ѵ",
            "ԝ",
            "×",
            "у",
            "ᴢ",
        }
    def __call__(self, text):
        words = []
        start_positions = []
        word = ""
        for i, c in enumerate(text):
            if c.isalnum() or c in EnglishWordSegmenter.HOMOGLYPHS:
                word += c
            elif c in "'-_*@" and len(word) > 0:
                # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the
                # word.
                word += c
            elif word:
                words.append(word)
                start_positions.append(i - len(word))
                word = ""
        if len(word) > 0:
            words.append(word)
            start_positions.append(len(text) - len(word))
        return words, start_positions


class FrenchWordSegmenter(EnglishWordSegmenter):
    # If you believe word segmenter for French should be different, please submit an issue or PR on Github.
    pass


class SpanishWordSegmenter(EnglishWordSegmenter):
    # If you believe word segmenter for Spanish should be different, please submit an issue or PR on Github.
    pass


class GermanWordSegmenter(EnglishWordSegmenter):
    # If you believe word segmenter for German should be different, please submit an issue or PR on Github.
    pass


class ItalianWordSegmenter(EnglishWordSegmenter):
    # If you believe word segmenter for Italian should be different, please submit an issue or PR on Github.
    pass


class KoreanWordSegmenter(WordSegmenter):
    """Segmenter for splitting Korean text into "words" (or more accurately,
    morphemes)."""

    # Set to `True` if we also return list of part-of-speech tags (for free).
    POS_TAGGING_INCLUDED = True

    def __init__(self):
        # `khaiii` is a Korean text analyzer released by Kakao (https://github.com/kakao/khaiii)
        custom_install_direction = (
            "Lazy module loader cannot find module named `khaiii`. "
            "This might be because TextAttack does not automatically install some optional dependencies. "
            "Please visit https://github.com/kakao/khaiii to install the Korean text analyzer Khaiii. "
        )
        khaiii = utils.LazyLoader(
            "khaiii", globals(), "khaiii", custom_direction=custom_install_direction
        )
        self.api = khaiii.KhaiiiApi()

    def __call__(self, text):
        """Accepts single string and returns a list of words, a list of start
        positions of each of the words.

        and a list of part-of-speech tag of each of the words.
        """
        khaiii_words = self.api.analyze(text)
        words = []
        start_positions = []
        pos_tags = []
        last_start = -1
        last_end = -1
        for word in khaiii_words:
            for morph in word.morphs:
                # Recover original form of text using position
                w = text[morph.begin : morph.begin + morph.length]
                if all([c in string.punctuation for c in w]):
                    # Skip punctuations
                    continue
                elif morph.begin >= last_start and morph.begin < last_end:
                    # This is necessary when some parts of word can be broken down into multiple morphemes.
                    # Ex) 안녕하세요 --> 안녕, 하, 시, 어요.
                    # Such breakdown would result in words being 안녕, 하, 세, 세요.
                    # We merge the last two by choosing the latest morph.
                    words[-1] = text[last_start : morph.begin + morph.length]
                    start_positions[-1] = last_start
                    pos_tags[-1] = pos_tags[-1] + "+" + morph.tag
                    last_end = morph.begin + morph.length
                else:
                    words.append(w)
                    start_positions.append(morph.begin)
                    pos_tags.append(morph.tag)
                    last_start = morph.begin
                    last_end = last_start + morph.length
        return words, start_positions, pos_tags

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        custom_install_direction = (
            "Lazy module loader cannot find module named `khaiii`. "
            "This might be because TextAttack does not automatically install some optional dependencies. "
            "Please visit https://github.com/kakao/khaiii to install the Korean text analyzer Khaiii. "
        )
        khaiii = utils.LazyLoader(
            "khaiii", globals(), "khaiii", custom_direction=custom_install_direction
        )
        self.api = khaiii.KhaiiiApi()



class ChineseWordSegmenter(WordSegmenter):
    """Segmenter for splitting Chinese text into "words" (or more accurately,
    morphemes)."""

    # Set to `True` if we also return list of part-of-speech tags (for free).
    POS_TAGGING_INCLUDED = True

    def __init__(self):
        stanza = utils.LazyLoader("stanza", globals(), "stanza")
        self.pipeline = stanza.Pipeline(lang="zh", processors="tokenize, pos", tokenize_no_ssplit=True)

    def __call__(self, text):
        """Accepts single string and returns a list of words, a list of start
        positions of each of the words.

        and a list of part-of-speech tag of each of the words.
        """
        
        text_list = text.split("\n")
        doc = self.pipeline(text_list)
        words = []
        start_positions = []
        pos_tags = []
        for sent in doc.sentences:
            for word in sent.words:
                # This strips punctuations
                words.append(word.text)
                start_positions.append(int(word.misc.split("|")[0][11:]))
                pos_tags.append(word.xpos)
        return words, start_positions, pos_tags

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        stanza = utils.LazyLoader("stanza", globals(), "stanza")
        self.pipeline = stanza.Pipeline(lang="zh", processors="tokenize, pos", tokenize_no_ssplit=True)
