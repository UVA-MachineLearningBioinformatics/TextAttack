from argparse import ArgumentDefaultsHelpFormatter, ArgumentError, ArgumentParser
import csv
import os
import time

import tqdm

import textattack
from textattack.commands import TextAttackCommand

COVERAGE_NAMES = {
    "perplexity": "textattack.coverage.PerplexityCoverage",
}


class CoverageCommand(TextAttackCommand):
    """The TextAttack coverage module:

    A command line parser to run test coverage from user provided
    specifications
    """

    def run(self, args):
        textattack.shared.utils.set_seed(args.random_seed)
        start_time = time.time()

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "coverage",
            help="measure coverage of testing data",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--model",
            "--m",
            help="model that is subject of testing",
            type=str,
            required=False,
            default=None,
        )
        parser.add_argument(
            "--dataset",
            "--d",
            help="name of the dataset to measure coverage",
            type=str,
            required=True,
            default=None,
        )
        parser.add_argument(
            "--coverage",
            "--c",
            help="name of coverage method",
            type=str,
            required=True,
            choices=AUGMENTATION_RECIPE_NAMES.keys(),
        )
        parser.add_argument(
            "--coverage-from-file",
            help="load `Coverage` from a file"
        )
        parser.add_argument(
            "--random-seed", default=42, type=int, help="random seed to set"
        )
        parser.set_defaults(func=AugmentCommand())
