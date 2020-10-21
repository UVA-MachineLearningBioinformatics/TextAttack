import logging

import torch
import transformers
from tqdm import tqdm

import textattack
from collections import defaultdict
from .coverage import ExtrinsicCoverage

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)



COVERAGE_MODEL_TYPES = ['bert-base-uncased']

class neuronCoverage(ExtrinsicCoverage):
		"""
		``neuronCoverage`` measures the neuron coverage acheived by a testset 
		Args:
				test_model(Union[str, torch.nn.Module]): name of the pretrained language model from `transformers`
						or the actual language model as a `torch.nn.Module` class. Default is "gpt2" from `transformers`.
				tokenizer (:obj:``, optional): If `test_model` is not a pretrained model from `transformers, need to provide
						the tokenizer here.
				max_seq_len (int): Maximum sequence length accepted by the model to be tested.  However, if you are using a pretrained model from `transformers`, this is handled
						automatically using information from `model.config`.
				threshold: threshold for marking a neuron as activated
				coarse_coverage(bool): if measure neuron coverage at the level of layer outputs 
		"""

		def __init__(
				self, test_model="bert-base-uncased-ag-news", tokenizer=None, max_seq_len=512, threshold=0.0, coarse_coverage = True):
				self.test_model = None
				self.coarse_coverage = coarse_coverage
				for available_model in COVERAGE_MODEL_TYPES:
					if available_model in test_model:
						self.model_type = available_model
						config = transformers.AutoConfig.from_pretrained(
								test_model, output_hidden_states = True
						)
						self.test_model = transformers.AutoModelForSequenceClassification.from_pretrained(test_model,config=config)   
						self.tokenizer = transformers.AutoTokenizer.from_pretrained(test_model, use_fast=True)
						self.max_seq_len = (
                			max_seq_len
                			if max_seq_len != -1
                			else self.test_model.config.n_positions
            )
				if self.test_model is None:
						raise ValueError(
								"`neuronCoverage` only accepts models in "+ ",".join(COVERAGE_MODEL_TYPES)
						)

				self.test_model.to(textattack.shared.utils.device)
				self.threshold = threshold
				self.test_model.eval()
				self.coverage_tracker = self._init_coverage()


		def _init_coverage(self):
			if self.model_type == 'bert-base-uncased':
				num_layers=13
				intermediate_hidden_state = 768
				
				
				if self.coarse_coverage:
					coverage_tracker = torch.zeros((num_layers, self.max_seq_len, intermediate_hidden_state), dtype=torch.bool)
				return coverage_tracker
					
		def _update_coarse_coverage(self, text):
			encodings = self.tokenizer(text, return_tensors="pt")
			if self.max_seq_len > 0:
					input_ids = encodings.input_ids[:, : self.max_seq_len]
					attention_mask = encodings.attention_mask[:, : self.max_seq_len]

			input_ids = input_ids.to(textattack.shared.utils.device)
			attention_mask = attention_mask.to(textattack.shared.utils.device)
			outputs = self.test_model(input_ids, attention_mask=attention_mask)
			classifier_outputs = outputs[0]
			for h_index, hidden_vector in enumerate(outputs[1]):
				
				self.coverage_tracker[h_index,:hidden_vector.size()[1]] = torch.where(hidden_vector[0,...]>self.threshold, torch.ones((hidden_vector.size()[1], self.coverage_tracker.size(2)), dtype=torch.bool).to(textattack.shared.utils.device) ,\
																											torch.zeros((hidden_vector.size()[1], self.coverage_tracker.size(2)), dtype=bool).to(textattack.shared.utils.device))
			

		def _compute_coverage(self):
			
			neuron_coverage = (torch.sum(self.coverage_tracker).item()/(1.0*self.coverage_tracker.numel()))
			return neuron_coverage

		def _update_coverage(self, text):
			if self.coarse_coverage:
				self._update_coarse_coverage(text)
			else:
				pass



		def __call__(self, testset):
				"""
				Returns neuron of `testset`
				Args:
						testset: Iterable of strings
				Returns:
						neuron coverage (float)
				"""
				for t in tqdm(testset):
								self._update_coverage(t)
				neuron_coverage = self._compute_coverage()
				return neuron_coverage

