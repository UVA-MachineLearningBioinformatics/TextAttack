from argparse import ArgumentParser

from textattack.commands import TextAttackCommand
class BenchmarkRecipeCommand(TextAttackCommand):
    """
    The TextAttack benchmark recipe module:
    
        A command line parser to benchmark a recipe from user specifications.
    """
    
    def run(self):
        raise NotImplementedError('cant benchmark yet')

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser("benchmark-recipe", help="Benchmark a model with TextAttack")