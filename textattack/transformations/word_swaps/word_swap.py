"""
Word Swap
============================================
Word swap transformations act by replacing some words in the input. Subclasses can implement the abstract ``WordSwap`` class by overriding ``self._get_replacement_words``

"""
import random
import string

from textattack.transformations import Transformation


class WordSwap(Transformation):
    """An abstract class that takes a sentence and transforms it by replacing
    some of its words.

    letters_to_insert (string): letters allowed for insertion into words
    (used by some char-based transformations)
    """

    def __init__(self, letters_to_insert=None):
        self.letters_to_insert = letters_to_insert
        if not self.letters_to_insert:
            self.letters_to_insert = string.ascii_letters

    def __call__(
        self,
        current_text,
        pre_transformation_constraints=[],
        indices_to_modify=None,
        shifted_idxs=False,
    ):
        transformed_texts = super().__call__(
            current_text,
            pre_transformation_constraints=pre_transformation_constraints,
            indices_to_modify=indices_to_modify,
            shifted_idxs=shifted_idxs,
        )
        # Filter transformed texts so that number of words stay constant.
        return [
            t
            for t in transformed_texts
            if t.num_words == current_text.num_words
        ]
        # a = []
        # for t in transformed_texts:
        #     if t.num_words == current_text.num_words:
        #         if -1 not in t.attack_attrs["original_index_map"]:
        #             a.append(t)
        #         else:
        #             print(t.attack_attrs["original_index_map"])
        #     else:
        #         print("doo")
        # return a
        # return transformed_texts

    def _get_replacement_words(self, word):
        """Returns a set of replacements given an input word. Must be overriden
        by specific word swap transformations.

        Args:
            word: The input word to find replacements for.
        """
        raise NotImplementedError()

    def _get_random_letter(self):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(self.letters_to_insert)

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue

                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts
