from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import textattack
from textattack.commands import TextAttackCommand
from textattack.commands.coverage.coverage_args import COVERAGE_NAMES
from textattack.commands.shared_args import add_dataset_args, add_model_args


class CoverageCommand(TextAttackCommand):
    """The TextAttack coverage module:

    A command line parser to run test coverage from user provided
    specifications
    """

    def run(self, args):
        textattack.shared.utils.set_seed(args.random_seed)

        from textattack.commands.coverage.run_coverage_parallel import (
            run as run_parallel,
        )
        from textattack.commands.coverage.run_coverage_single_threaded import (
            run as run_single_threaded,
        )

        if args.parallel:
            run_parallel(args)
        else:
            run_single_threaded(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "coverage",
            help="measure coverage of testing data",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--coverage",
            "--c",
            help="name of coverage method",
            type=str,
            required=True,
            choices=COVERAGE_NAMES.keys(),
        )
        parser.add_argument("--coverage-from-file", help="load `Coverage` from a file")

        add_model_args(parser)
        add_dataset_args(parser)

        parser.add_argument("--random-seed", default=765, type=int)
        parser.add_argument("--parallel", default=False, type=bool)

        parser.set_defaults(func=CoverageCommand())
