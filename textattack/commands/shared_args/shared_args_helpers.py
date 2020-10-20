import argparse
import json
import os

import textattack
from textattack.shared.utils import ARGS_SPLIT_TOKEN, load_module_from_file

from .model_and_dataset_args import (
    HUGGINGFACE_DATASET_BY_MODEL,
    TEXTATTACK_DATASET_BY_MODEL,
)


def add_model_args(parser):
    """Adds model-related arguments to an argparser.

    This is useful because we want to load pretrained models using
    multiple different parsers that share these, but not all, arguments.
    """
    model_group = parser.add_mutually_exclusive_group()

    model_names = list(HUGGINGFACE_DATASET_BY_MODEL.keys()) + list(
        TEXTATTACK_DATASET_BY_MODEL.keys()
    )
    model_group.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help='Name of or path to a pre-trained model to attack. Usage: "--model {model}:{arg_1}={value_1},{arg_3}={value_3},...". Choices: '
        + str(model_names),
    )
    model_group.add_argument(
        "--model-from-file",
        type=str,
        required=False,
        help="File of model and tokenizer to import.",
    )
    model_group.add_argument(
        "--model-from-huggingface",
        type=str,
        required=False,
        help="huggingface.co ID of pre-trained model to load",
    )


def add_dataset_args(parser):
    """Adds dataset-related arguments to an argparser.

    This is useful because we want to load pretrained models using
    multiple different parsers that share these, but not all, arguments.
    """
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset-from-huggingface",
        type=str,
        required=False,
        default=None,
        help="Dataset to load from `datasets` repository.",
    )
    dataset_group.add_argument(
        "--dataset-from-file",
        type=str,
        required=False,
        default=None,
        help="Dataset to load from a file.",
    )
    parser.add_argument(
        "--shuffle",
        type=eval,
        required=False,
        choices=[True, False],
        default="True",
        help="Randomly shuffle the data before attacking",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        required=False,
        default="5",
        help="The number of examples to process.",
    )

    parser.add_argument(
        "--num-examples-offset",
        "-o",
        type=int,
        required=False,
        default=0,
        help="The offset to start at in the dataset.",
    )


def parse_model_from_args(args):
    if args.model_from_file:
        # Support loading the model from a .py file where a model wrapper
        # is instantiated.
        colored_model_name = textattack.shared.utils.color_text(
            args.model_from_file, color="blue", method="ansi"
        )
        textattack.shared.logger.info(
            f"Loading model and tokenizer from file: {colored_model_name}"
        )
        if ARGS_SPLIT_TOKEN in args.model_from_file:
            model_file, model_name = args.model_from_file.split(ARGS_SPLIT_TOKEN)
        else:
            _, model_name = args.model_from_file, "model"
        try:
            model_module = load_module_from_file(args.model_from_file)
        except Exception:
            raise ValueError(f"Failed to import file {args.model_from_file}")
        try:
            model = getattr(model_module, model_name)
        except AttributeError:
            raise AttributeError(
                f"``{model_name}`` not found in module {args.model_from_file}"
            )

        if not isinstance(model, textattack.models.wrappers.ModelWrapper):
            raise TypeError(
                "Model must be of type "
                f"``textattack.models.ModelWrapper``, got type {type(model)}"
            )
    elif (args.model in HUGGINGFACE_DATASET_BY_MODEL) or args.model_from_huggingface:
        # Support loading models automatically from the HuggingFace model hub.
        import transformers

        model_name = (
            HUGGINGFACE_DATASET_BY_MODEL[args.model][0]
            if (args.model in HUGGINGFACE_DATASET_BY_MODEL)
            else args.model_from_huggingface
        )

        if ARGS_SPLIT_TOKEN in model_name:
            model_class, model_name = model_name
            model_class = eval(f"transformers.{model_class}")
        else:
            model_class, model_name = (
                transformers.AutoModelForSequenceClassification,
                model_name,
            )
        colored_model_name = textattack.shared.utils.color_text(
            model_name, color="blue", method="ansi"
        )
        textattack.shared.logger.info(
            f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
        )
        model = model_class.from_pretrained(model_name)
        tokenizer = textattack.models.tokenizers.AutoTokenizer(model_name)
        model = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer, batch_size=args.model_batch_size
        )
    elif args.model in TEXTATTACK_DATASET_BY_MODEL:
        # Support loading TextAttack pre-trained models via just a keyword.
        model_path, _ = TEXTATTACK_DATASET_BY_MODEL[args.model]
        model = textattack.shared.utils.load_textattack_model_from_path(
            args.model, model_path
        )
        # Choose the approprate model wrapper (based on whether or not this is
        # a HuggingFace model).
        if isinstance(
            model, textattack.models.helpers.BERTForClassification
        ) or isinstance(model, textattack.models.helpers.T5ForTextToText):
            model = textattack.models.wrappers.HuggingFaceModelWrapper(
                model, model.tokenizer, batch_size=args.model_batch_size
            )
        else:
            model = textattack.models.wrappers.PyTorchModelWrapper(
                model, model.tokenizer, batch_size=args.model_batch_size
            )
    elif args.model and os.path.exists(args.model):
        # Support loading TextAttack-trained models via just their folder path.
        # If `args.model` is a path/directory, let's assume it was a model
        # trained with textattack, and try and load it.
        model_args_json_path = os.path.join(args.model, "train_args.json")
        if not os.path.exists(model_args_json_path):
            raise FileNotFoundError(
                f"Tried to load model from path {args.model} - could not find train_args.json."
            )
        model_train_args = json.loads(open(model_args_json_path).read())
        if model_train_args["model"] not in {"cnn", "lstm"}:
            # for huggingface models, set args.model to the path of the model
            model_train_args["model"] = args.model
        num_labels = model_train_args["num_labels"]
        from textattack.commands.train_model.train_args_helpers import model_from_args

        model = model_from_args(
            argparse.Namespace(**model_train_args),
            num_labels,
            model_path=args.model,
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(
            model, model.tokenizer, batch_size=args.model_batch_size
        )
    else:
        raise ValueError(f"Error: unsupported TextAttack model {args.model}")

    return model


def parse_dataset_from_args(args):
    # Automatically detect dataset for huggingface & textattack models.
    # This allows us to use the --model shortcut without specifying a dataset.
    if args.model in HUGGINGFACE_DATASET_BY_MODEL:
        _, args.dataset_from_huggingface = HUGGINGFACE_DATASET_BY_MODEL[args.model]
    elif args.model in TEXTATTACK_DATASET_BY_MODEL:
        _, dataset = TEXTATTACK_DATASET_BY_MODEL[args.model]
        if dataset[0].startswith("textattack"):
            # unsavory way to pass custom dataset classes
            # ex: dataset = ('textattack.datasets.translation.TedMultiTranslationDataset', 'en', 'de')
            dataset = eval(f"{dataset[0]}")(*dataset[1:])
            return dataset
        else:
            args.dataset_from_huggingface = dataset
    # Automatically detect dataset for models trained with textattack.
    elif args.model and os.path.exists(args.model):
        model_args_json_path = os.path.join(args.model, "train_args.json")
        if not os.path.exists(model_args_json_path):
            raise FileNotFoundError(
                f"Tried to load model from path {args.model} - could not find train_args.json."
            )
        model_train_args = json.loads(open(model_args_json_path).read())
        try:
            if ARGS_SPLIT_TOKEN in model_train_args["dataset"]:
                name, subset = model_train_args["dataset"].split(ARGS_SPLIT_TOKEN)
            else:
                name, subset = model_train_args["dataset"], None
            args.dataset_from_huggingface = (
                name,
                subset,
                model_train_args["dataset_dev_split"],
            )
        except KeyError:
            raise KeyError(
                f"Tried to load model from path {args.model} but can't initialize dataset from train_args.json."
            )

    # Get dataset from args.
    if args.dataset_from_file:
        textattack.shared.logger.info(
            f"Loading model and tokenizer from file: {args.model_from_file}"
        )
        if ARGS_SPLIT_TOKEN in args.dataset_from_file:
            dataset_file, dataset_name = args.dataset_from_file.split(ARGS_SPLIT_TOKEN)
        else:
            dataset_file, dataset_name = args.dataset_from_file, "dataset"
        try:
            dataset_module = load_module_from_file(dataset_file)
        except Exception:
            raise ValueError(
                f"Failed to import dataset from file {args.dataset_from_file}"
            )
        try:
            dataset = getattr(dataset_module, dataset_name)
        except AttributeError:
            raise AttributeError(
                f"``dataset`` not found in module {args.dataset_from_file}"
            )
    elif args.dataset_from_huggingface:
        dataset_args = args.dataset_from_huggingface
        if isinstance(dataset_args, str):
            if ARGS_SPLIT_TOKEN in dataset_args:
                dataset_args = dataset_args.split(ARGS_SPLIT_TOKEN)
            else:
                dataset_args = (dataset_args,)
        dataset = textattack.datasets.HuggingFaceDataset(
            *dataset_args, shuffle=args.shuffle
        )
        dataset.examples = dataset.examples[args.num_examples_offset :]
    else:
        raise ValueError("Must supply pretrained model or dataset")
    return dataset
