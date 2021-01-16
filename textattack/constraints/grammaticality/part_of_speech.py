"""
Part of Speech Constraint
--------------------------
"""

from textattack.constraints import Constraint
from textattack.shared.validators import transformation_consists_of_word_swaps


class PartOfSpeech(Constraint):
    """Constraints word swaps to only swap words with the same part of speech.
    Uses the NLTK universal part-of-speech tagger by default. An implementation
    of `<https://arxiv.org/abs/1907.11932>`_ adapted from
    `<https://github.com/jind11/TextFooler>`_.

    Args:
        pos_tagger (textattack.shared.PosTagger): Part-of-speech tagger to use. Default is Flair's POS tagger using universal tag set.
        allow_verb_noun_swap (bool): If `True`, allow verbs to be swapped with nouns and vice versa.
        compare_against_original (bool): If `True`, compare against the original text.
            Otherwise, compare against the most recent text.
    """

    def __init__(
        self,
        allow_verb_noun_swap=True,
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        self.allow_verb_noun_swap = allow_verb_noun_swap

    def _can_replace_pos(self, pos_a, pos_b):
        return (pos_a == pos_b) or (
            self.allow_verb_noun_swap and set([pos_a, pos_b]) <= set(["NOUN", "VERB"])
        )

    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )

        for i in indices:
            ref_pos = reference_text.pos_tags[i]
            replace_pos = transformed_text.pos_tags[i]
            if not self._can_replace_pos(ref_pos, replace_pos):
                return False

        return True

    def check_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        return [
            "allow_verb_noun_swap",
        ] + super().extra_repr_keys()
