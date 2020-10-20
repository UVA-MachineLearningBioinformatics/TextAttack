import logging

import torch
import transformers
from tqdm import tqdm

import textattack

from .coverage import ExtrinsicCoverage

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class PerplexityCoverage(ExtrinsicCoverage):
    """
    ``PerplexityCoverage`` meausures the average perplexity of a given test datsaet using a language model
    Args:
        language_model(Union[str, torch.nn.Module]): name of the pretrained language model from `transformers`
            or the actual language model as a `torch.nn.Module` class. Default is "gpt2" from `transformers`.
        tokenizer (:obj:``, optional): If `language_model` is not a pretrained model from `transformers, need to provide
            the tokenizer here.
        max_seq_len (int): Maximum sequence length accepted by the language model. Please set this if you're using
            fixed-length language model. However, if you are using a pretrained model from `transformers`, this is handled
            automatically using information from `model.config`.
    """

    def __init__(
        self, language_model="gpt2", tokenizer=None, max_seq_len=-1):
        if isinstance(language_model, str):
            self.language_model = transformers.AutoModelForCausalLM.from_pretrained(
                language_model
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(language_model, use_fast=True)
            self.max_seq_len = (
                max_seq_len
                if max_seq_len != -1
                else self.language_model.config.n_positions
            )
            self._hf = True
        elif isinstance(language_model, torch.nn.Module):
            if tokenizer is None:
                raise ValueError(
                    "`tokenizer` must be provided if `language_model` is a torch.nn.Module class"
                )
            self.language_model = language_model
            self.tokenzier = tokenizer
            self.max_seq_len = max_seq_len
            self._hf = False
        else:
            raise ValueError(
                "`PerplexityCoverage` only accepts models from `transformers` package or `torch.nn.Module` models."
            )

        self.language_model.to(textattack.shared.utils.device)
        self.language_model.eval()

    def _hf_calc_perplexity(self, text):
        """Calculate the perplexity of `text` for Huggingface models.
        Args:
            text (str): text to calculate perplexity of.
        Returns:
            perplexity of `text` as float.
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        if self.max_seq_len > 0:
            input_ids = encodings.input_ids[:, : self.max_seq_len]
            attention_mask = encodings.attention_mask[:, : self.max_seq_len]

        input_ids = input_ids.to(textattack.shared.utils.device)
        attention_mask = attention_mask.to(textattack.shared.utils.device)
        
        loss = self.language_model(input_ids, attention_mask=attention_mask, labels=input_ids)[0]
        ppl = torch.exp(loss).item()
        if encodings.input_ids.shape[1] > self.max_seq_len and self.max_seq_len > 0:
            ppl = ppl * input_ids.shape[1]
            k = 0
            for i in range(self.max_seq_len+1, encodings.input_ids.shape[1]):
                input_ids = encodings.input_ids[:, i-self.max_seq_len:i].to(textattack.shared.utils.device)
                attention_mask = encodings.attention_mask[:, i-self.max_seq_len:i].to(textattack.shared.utils.device)
                label_ids = input_ids.clone()
                label_ids[:, :-1] = -100
                label_ids.to(textattack.shared.utils.device)
                loss = self.language_model(input_ids, attention_mask=attention_mask, labels=label_ids)[0]
                ppl += torch.exp(loss).item()
                k += 1

            ppl = ppl / (input_ids.shape[1] + k)

        return ppl

    def __call__(self, testset):
        """
        Returns average perplexity of `testset`
        Args:
            testset: Iterable of strings
        Returns:
            average perplexity (float)
        """
        total_ppl = 0
        for t in tqdm(testset):
            if self._hf:
                total_ppl += self._hf_calc_perplexity(t)
            else:
                #TODO add `calc_perlexity` for regular torch.nn.Module models
                pass 
        avg_ppl = total_ppl / len(testset)
        return avg_ppl
