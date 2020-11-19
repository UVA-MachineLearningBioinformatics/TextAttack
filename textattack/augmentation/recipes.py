"""
Augmenter Recipes:
===================

Transformations and constraints can be used for simple NLP data augmentations. Here is a list of recipes for NLP data augmentations

"""
import random

import textattack

from . import Augmenter

DEFAULT_CONSTRAINTS = [
    textattack.constraints.pre_transformation.RepeatModification(),
    textattack.constraints.pre_transformation.StopwordModification(),
]


class CoverageGuidedAugmenter(Augmenter):
    """An implementation of Easy Data Augmentation, which combines:

    - WordNet synonym replacement
                    - Randomly replace words with their synonyms.
    - Word deletion
                    - Randomly remove words from the sentence.
    - Word order swaps
                    - Randomly swap the position of words in the sentence.
    - Random synonym insertion
                    - Insert a random synonym of a random word at a random location.

    in one augmentation method.

    "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" (Wei and Zou, 2019)
    https://arxiv.org/abs/1901.11196
    """

    def __init__(
        self,
        test_model,
        K=2,
        iterations=10,
        transformations_per_example=1,
        outfile="aug.csv",
        savedir="./",
        beam_size=-1,
        epsilon=0.0001,
        key="text",
        language_model="gpt2",
        max_seq_len=-1,
        pct_words_to_swap=0.1,
        threshold=0.5,
        select_swap_parameter=False,
        beam=1,
        perplexity_tolerance=0.5,
    ):
        assert (
            pct_words_to_swap >= 0.0 and pct_words_to_swap <= 1.0
        ), "pct_words_to_swap must be in [0., 1.]"
        assert (
            transformations_per_example > 0
        ), "transformations_per_example must be a positive integer"
        self.pct_words_to_swap = pct_words_to_swap
        self.select_swap_parameter = select_swap_parameter

        self.transformations_per_example = transformations_per_example
        self.test_model = test_model
        self.nc_test_model = neuronCoverage(
            self.test_model, threshold=threshold, max_seq_len=max_seq_len
        )
        self.perplexity_calculator = PerplexityCoverage(
            language_model=language_model,
            tokenizer=self.test_model,
            max_seq_len=max_seq_len,
        )
        self.K = K
        self.iterations = iterations
        self.perplexity_tolerance = perplexity_tolerance
        self.available_transformations = {
            textattack.transformations.WordDeletion,
            textattack.transformations.WordSwapEmbedding,
            textattack.transformations.WordSwapChangeLocation,
            textattack.transformations.WordSwapChangeName,
            textattack.transformations.WordSwapChangeNumber,
            textattack.transformations.WordSwapContract,
            textattack.transformations.WordSwapExtend,
            textattack.transformations.WordSwapHomoglyphSwap,
            textattack.transformations.WordSwapMaskedLM,
            textattack.transformations.WordSwapNeighboringCharacterSwap,
            textattack.transformations.WordSwapRandomCharacterDeletion,
            textattack.transformations.WordSwapRandomCharacterInsertion,
            textattack.transformations.WordSwapRandomCharacterSubstitution,
            textattack.transformations.RandomSwap,
            textattack.transformations.WordSwapWordNet,
        }
        self.epsilon = epsilon
        self.beam = beam

    def augment_many(self, seed_texts, labels):
        """
        seed_texts: list of dataset samples

        """
        flag = 0  # for later to resume from the middle as this is time consuming and can fail

        if not isinstance(seed_texts, list):
            seed_texts = [seed_texts]

        if not isinstance(labels, list):
            labels = [labels]

        original_neuron_coverage = self.nc_test_model(seed_texts)
        print("original neuron coverage:", original_neuron_coverage)
        current_neuron_coverage = original_neuron_coverage

        text_queue = deque()
        text_perp = deque()
        text_y_queue = deque()
        text_stack = []
        text_y_stack = []
        generated_test = []
        generated_test_label = []
        generated = 0

        cache = deque()

        for seed_text, label in zip(seed_texts, labels):

            # if previous example is done, add new example
            text_queue.append(seed_text)  # text_queue stores all the input seed texts
            text_perp.append(self.perplexity_calculator([seed_text]))
            text_y_queue.append(label)

        while len(text_queue) > 0:
            # till we still have samples to transform
            current_seed_text = text_queue[0]
            current_seed_label = text_y_queue[0]
            current_seed_perp = text_perp[0]
            print(f"{len(text_queue)} samples are left.")
            # the prediction for this sample

            baseline_yhat = torch.max(
                self.nc_test_model._eval(current_seed_text)[0], dim=-1
            )[1]
            # ground truth of sample
            # truth = current_seed_text[1]

            num_generated = (
                0  # to limit the number of test samples generated per sample
            )
            if len(text_stack) == 0:
                # add current seed text to stack
                text_stack.append(current_seed_text)
                # prediction of this sample
                text_y_stack.append(baseline_yhat)

            while len(text_stack) > 0:

                # take seed text or latest transformed sample

                current_text = text_stack[-1]

                new_generated = False
                for i in range(self.iterations):
                    print("on iteration: ", i)
                    # sample K transformations
                    sampled_transformations = random.sample(
                        list(self.available_transformations), self.K
                    )
                    print("sampled transformations:", sampled_transformations)

                    tid = []
                    if self.select_swap_parameter:
                        pct_words_to_swap = self.pct_words_to_swap
                    else:
                        pct_words_to_swap = random.uniform(0.0, 1.0)
                        print("pct words to swap: ", pct_words_to_swap)
                    for transformation in sampled_transformations:
                        if isinstance(
                            transformation, textattack.transformations.WordSwapEmbedding
                        ):
                            constraints = DEFAULT_CONSTRAINTS + [
                                textattack.constraints.semantics.WordEmbeddingDistance(
                                    min_cos_sim=0.8
                                )
                            ]
                        else:
                            constraints = DEFAULT_CONSTRAINTS

                        tid.append(
                            Augmenter(
                                transformation=transformation(),
                                constraints=constraints,
                                pct_words_to_swap=pct_words_to_swap,
                                transformations_per_example=self.transformations_per_example,
                            )
                        )
                    print("sampled built augmenters:")
                    print(tid)
                    if len(cache) > 0:
                        # for two transformations, preserve the first transformation
                        print("cache saved good transformations")

                        for b in range(self.beam):
                            tid[b] = cache.popleft()
                        print(tid)

                    print("generating sample!")
                    new_text = copy.deepcopy(current_text)
                    # transform sample

                    for j in range(self.K):
                        print("sampling from", tid[j])
                        new_text = tid[j].augment(new_text)[0]
                        print("augmented text!")
                        print(new_text)

                    # check if this sample increases coverage
                    temp_nc = copy.deepcopy(self.nc_test_model)
                    # measure neuron coverage
                    new_neuron_coverage = temp_nc([new_text])
                    print(
                        "original perplexity:",
                        current_seed_perp,
                        "new perplexity:",
                        self.perplexity_calculator([new_text]),
                    )

                    if abs(self.perplexity_calculator([new_text])) > abs(
                        current_seed_perp
                    ):
                        augment_perplexity = abs(
                            abs(self.perplexity_calculator([new_text]))
                            / abs(current_seed_perp)
                        )
                    else:
                        augment_perplexity = abs(
                            abs(current_seed_perp)
                            / abs(self.perplexity_calculator([new_text]))
                        )
                    # prediction for augmented text
                    print("ratio of perpelxity: ", augment_perplexity)
                    augment_yhat = torch.max(
                        self.nc_test_model._eval(new_text)[0], dim=-1
                    )[1]
                    print(
                        "new neuron coverage:",
                        new_neuron_coverage,
                        " original:",
                        current_neuron_coverage,
                    )
                    print(
                        "does this increase neuron coverage?",
                        (new_neuron_coverage - current_neuron_coverage) > self.epsilon,
                    )
                    print(
                        "is this a good sentence?",
                        augment_perplexity < self.perplexity_tolerance,
                    )
                    if (
                        new_neuron_coverage - current_neuron_coverage
                    ) > self.epsilon and (
                        augment_perplexity < self.perplexity_tolerance
                    ):
                        # append the transformations to the trandformation cache
                        cache.extend([t for t in tid])
                        generated += 1
                        num_generated += 1
                        print("Generated text increases coverage, generated test no: ")
                        print(generated)
                        print("augmenting using: ", new_text)

                        # text_stack stores the generated test set, for further transformations
                        text_stack.append(new_text)
                        # save the  text inputs and ground truths
                        generated_test.append(new_text)
                        generated_test_label.append(current_seed_label)
                        # the current prediction, may use it later to add constraints
                        text_y_stack.append(baseline_yhat)
                        # valid test input, so update the model's neuron coverage
                        self.nc_test_model._update_coverage(new_text)
                        # treat this as the new current coverage
                        p = self.nc_test_model._compute_coverage()
                        current_neuron_coverage = p
                        print("New Coverage: ", current_neuron_coverage)
                        new_generated = True
                        if current_neuron_coverage > 0.9999:
                            return (
                                generated_test,
                                generated_test_label,
                                [original_neuron_coverage, current_neuron_coverage],
                            )

                        break

                    else:
                        print("Generated text does not increase coverage.")
                if not new_generated:
                    # this input is completely processed, remove
                    text_stack.pop()
                    text_y_stack.pop()

            text_queue.popleft()
            text_y_queue.popleft()
            text_perp.popleft()
        return (
            generated_test,
            generated_test_label,
            [original_neuron_coverage, current_neuron_coverage],
        )

    def augment(self, text, labels=None):
        generated_test, generated_test_label = self.augment_many(text, labels)
        return generated_test, generated_test_label

    def __repr__(self):
        return "CoverageAugmenter"


class EasyDataAugmenter(Augmenter):
    """An implementation of Easy Data Augmentation, which combines:

    - WordNet synonym replacement
        - Randomly replace words with their synonyms.
    - Word deletion
        - Randomly remove words from the sentence.
    - Word order swaps
        - Randomly swap the position of words in the sentence.
    - Random synonym insertion
        - Insert a random synonym of a random word at a random location.

    in one augmentation method.

    "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" (Wei and Zou, 2019)
    https://arxiv.org/abs/1901.11196
    """

    def __init__(self, pct_words_to_swap=0.1, transformations_per_example=4):
        assert (
            pct_words_to_swap >= 0.0 and pct_words_to_swap <= 1.0
        ), "pct_words_to_swap must be in [0., 1.]"
        assert (
            transformations_per_example > 0
        ), "transformations_per_example must be a positive integer"
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example
        n_aug_each = max(transformations_per_example // 4, 1)

        self.synonym_replacement = WordNetAugmenter(
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=n_aug_each,
        )
        self.random_deletion = DeletionAugmenter(
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=n_aug_each,
        )
        self.random_swap = SwapAugmenter(
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=n_aug_each,
        )
        self.random_insertion = SynonymInsertionAugmenter(
            pct_words_to_swap=pct_words_to_swap, transformations_per_example=n_aug_each
        )

    def augment(self, text):
        augmented_text = []
        augmented_text += self.synonym_replacement.augment(text)
        augmented_text += self.random_deletion.augment(text)
        augmented_text += self.random_swap.augment(text)
        augmented_text += self.random_insertion.augment(text)
        random.shuffle(augmented_text)
        return augmented_text[: self.transformations_per_example]

    def __repr__(self):
        return "EasyDataAugmenter"


class SwapAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import RandomSwap

        transformation = RandomSwap()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class SynonymInsertionAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import RandomSynonymInsertion

        transformation = RandomSynonymInsertion()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class WordNetAugmenter(Augmenter):
    """Augments text by replacing with synonyms from the WordNet thesaurus."""

    def __init__(self, **kwargs):
        from textattack.transformations import WordSwapWordNet

        transformation = WordSwapWordNet()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class DeletionAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import WordDeletion

        transformation = WordDeletion()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class EmbeddingAugmenter(Augmenter):
    """Augments text by transforming words with their embeddings."""

    def __init__(self, **kwargs):
        from textattack.transformations import WordSwapEmbedding

        transformation = WordSwapEmbedding(
            max_candidates=50, embedding_type="paragramcf"
        )
        from textattack.constraints.semantics import WordEmbeddingDistance

        constraints = DEFAULT_CONSTRAINTS + [WordEmbeddingDistance(min_cos_sim=0.8)]
        super().__init__(transformation, constraints=constraints, **kwargs)


class CharSwapAugmenter(Augmenter):
    """Augments words by swapping characters out for other characters."""

    def __init__(self, **kwargs):
        from textattack.transformations import (
            CompositeTransformation,
            WordSwapNeighboringCharacterSwap,
            WordSwapRandomCharacterDeletion,
            WordSwapRandomCharacterInsertion,
            WordSwapRandomCharacterSubstitution,
        )

        transformation = CompositeTransformation(
            [
                # (1) Swap: Swap two adjacent letters in the word.
                WordSwapNeighboringCharacterSwap(),
                # (2) Substitution: Substitute a letter in the word with a random letter.
                WordSwapRandomCharacterSubstitution(),
                # (3) Deletion: Delete a random letter from the word.
                WordSwapRandomCharacterDeletion(),
                # (4) Insertion: Insert a random letter in the word.
                WordSwapRandomCharacterInsertion(),
            ]
        )
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class CheckListAugmenter(Augmenter):
    """Augments words by using the transformation methods provided by CheckList
    INV testing, which combines:

    - Name Replacement
    - Location Replacement
    - Number Alteration
    - Contraction/Extension

    "Beyond Accuracy: Behavioral Testing of NLP models with CheckList" (Ribeiro et al., 2020)
    https://arxiv.org/abs/2005.04118
    """

    def __init__(self, **kwargs):
        from textattack.transformations import (
            CompositeTransformation,
            WordSwapChangeLocation,
            WordSwapChangeName,
            WordSwapChangeNumber,
            WordSwapContract,
            WordSwapExtend,
        )

        transformation = CompositeTransformation(
            [
                WordSwapChangeNumber(),
                WordSwapChangeLocation(),
                WordSwapChangeName(),
                WordSwapExtend(),
                WordSwapContract(),
            ]
        )

        constraints = [DEFAULT_CONSTRAINTS[0]]

        super().__init__(transformation, constraints=constraints, **kwargs)
