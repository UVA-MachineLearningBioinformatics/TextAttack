import os
import pickle

import numpy as np

from textattack.shared import (
    EnglishPosTagger,
    EnglishWordSegmenter,
    WordEmbedding,
    utils,
)

from .language import LanguageResource


class EnglishResource(LanguageResource):

    """Singleton class responsible for provisioning default resources for
    different languages."""

    @property
    def word_segmenter(self):
        return EnglishWordSegmenter()

    @property
    def masked_lm_name(self):
        return "bert-base-uncased"

    @property
    def word_embedding(self):
        """Returns a prebuilt counter-fitted GLOVE word embedding proposed by
        "Counter-fitting Word Vectors to Linguistic Constraints" (Mrkšić et
        al., 2016)"""
        if not hasattr(self, "_word_embedding"):
            word_embeddings_folder = "paragramcf"
            word_embeddings_file = "paragram.npy"
            word_list_file = "wordlist.pickle"
            mse_dist_file = "mse_dist.p"
            cos_sim_file = "cos_sim.p"
            nn_matrix_file = "nn.npy"

            # Download embeddings if they're not cached.
            word_embeddings_folder = os.path.join(
                WordEmbedding.PATH, word_embeddings_folder
            )
            word_embeddings_folder = utils.download_if_needed(word_embeddings_folder)
            # Concatenate folder names to create full path to files.
            word_embeddings_file = os.path.join(
                word_embeddings_folder, word_embeddings_file
            )
            word_list_file = os.path.join(word_embeddings_folder, word_list_file)
            mse_dist_file = os.path.join(word_embeddings_folder, mse_dist_file)
            cos_sim_file = os.path.join(word_embeddings_folder, cos_sim_file)
            nn_matrix_file = os.path.join(word_embeddings_folder, nn_matrix_file)

            # loading the files
            embedding_matrix = np.load(word_embeddings_file)
            word2index = np.load(word_list_file, allow_pickle=True)
            index2word = {}
            for word, index in word2index.items():
                index2word[index] = word
            nn_matrix = np.load(nn_matrix_file)

            embedding = WordEmbedding(
                embedding_matrix, word2index, index2word, nn_matrix
            )

            with open(mse_dist_file, "rb") as f:
                mse_dist_mat = pickle.load(f)
            with open(cos_sim_file, "rb") as f:
                cos_sim_mat = pickle.load(f)

            embedding._mse_dist_mat = mse_dist_mat
            embedding._cos_sim_mat = cos_sim_mat

            self._word_embedding = embedding

        return self._word_embedding

    @property
    def wordnet_lang_code(self):
        return "eng"

    @property
    def pos_tagger(self):
        return EnglishPosTagger()

    @property
    def ner_tagger(self):
        raise NotImplementedError()
