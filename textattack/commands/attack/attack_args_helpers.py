"""

TextAttack Command Helpers for Attack
------------------------------------------

"""

import copy
import os
import time

import textattack
from textattack.commands.shared_args import parse_model_from_args
from textattack.shared.utils import ARGS_SPLIT_TOKEN, load_module_from_file

from .attack_args import (
    ATTACK_RECIPE_NAMES,
    BLACK_BOX_TRANSFORMATION_CLASS_NAMES,
    CONSTRAINT_CLASS_NAMES,
    GOAL_FUNCTION_CLASS_NAMES,
    SEARCH_METHOD_CLASS_NAMES,
    WHITE_BOX_TRANSFORMATION_CLASS_NAMES,
)


def parse_transformation_from_args(args, model_wrapper):
    transformation_name = args.transformation
    if ARGS_SPLIT_TOKEN in transformation_name:
        transformation_name, params = transformation_name.split(ARGS_SPLIT_TOKEN)

        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model_wrapper.model, {params})"
            )
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}({params})"
            )
        else:
            raise ValueError(f"Error: unsupported transformation {transformation_name}")
    else:
        if transformation_name in WHITE_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{WHITE_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}(model_wrapper.model)"
            )
        elif transformation_name in BLACK_BOX_TRANSFORMATION_CLASS_NAMES:
            transformation = eval(
                f"{BLACK_BOX_TRANSFORMATION_CLASS_NAMES[transformation_name]}()"
            )
        else:
            raise ValueError(f"Error: unsupported transformation {transformation_name}")
    return transformation


def parse_goal_function_from_args(args, model):
    goal_function = args.goal_function
    if ARGS_SPLIT_TOKEN in goal_function:
        goal_function_name, params = goal_function.split(ARGS_SPLIT_TOKEN)
        if goal_function_name not in GOAL_FUNCTION_CLASS_NAMES:
            raise ValueError(f"Error: unsupported goal_function {goal_function_name}")
        goal_function = eval(
            f"{GOAL_FUNCTION_CLASS_NAMES[goal_function_name]}(model, {params})"
        )
    elif goal_function in GOAL_FUNCTION_CLASS_NAMES:
        goal_function = eval(f"{GOAL_FUNCTION_CLASS_NAMES[goal_function]}(model)")
    else:
        raise ValueError(f"Error: unsupported goal_function {goal_function}")
    goal_function.query_budget = args.query_budget
    goal_function.model_batch_size = args.model_batch_size
    goal_function.model_cache_size = args.model_cache_size
    return goal_function


def parse_constraints_from_args(args):
    if not args.constraints:
        return []

    _constraints = []
    for constraint in args.constraints:
        if ARGS_SPLIT_TOKEN in constraint:
            constraint_name, params = constraint.split(ARGS_SPLIT_TOKEN)
            if constraint_name not in CONSTRAINT_CLASS_NAMES:
                raise ValueError(f"Error: unsupported constraint {constraint_name}")
            _constraints.append(
                eval(f"{CONSTRAINT_CLASS_NAMES[constraint_name]}({params})")
            )
        elif constraint in CONSTRAINT_CLASS_NAMES:
            _constraints.append(eval(f"{CONSTRAINT_CLASS_NAMES[constraint]}()"))
        else:
            raise ValueError(f"Error: unsupported constraint {constraint}")

    return _constraints


def parse_attack_from_args(args):
    model = parse_model_from_args(args)
    if args.recipe:
        if ARGS_SPLIT_TOKEN in args.recipe:
            recipe_name, params = args.recipe.split(ARGS_SPLIT_TOKEN)
            if recipe_name not in ATTACK_RECIPE_NAMES:
                raise ValueError(f"Error: unsupported recipe {recipe_name}")
            recipe = eval(f"{ATTACK_RECIPE_NAMES[recipe_name]}.build(model, {params})")
        elif args.recipe in ATTACK_RECIPE_NAMES:
            recipe = eval(f"{ATTACK_RECIPE_NAMES[args.recipe]}.build(model)")
        else:
            raise ValueError(f"Invalid recipe {args.recipe}")
        recipe.goal_function.query_budget = args.query_budget
        recipe.goal_function.model_batch_size = args.model_batch_size
        recipe.goal_function.model_cache_size = args.model_cache_size
        recipe.constraint_cache_size = args.constraint_cache_size
        return recipe
    elif args.attack_from_file:
        if ARGS_SPLIT_TOKEN in args.attack_from_file:
            attack_file, attack_name = args.attack_from_file.split(ARGS_SPLIT_TOKEN)
        else:
            attack_file, attack_name = args.attack_from_file, "attack"
        attack_module = load_module_from_file(attack_file)
        if not hasattr(attack_module, attack_name):
            raise ValueError(
                f"Loaded `{attack_file}` but could not find `{attack_name}`."
            )
        attack_func = getattr(attack_module, attack_name)
        return attack_func(model)
    else:
        goal_function = parse_goal_function_from_args(args, model)
        transformation = parse_transformation_from_args(args, model)
        constraints = parse_constraints_from_args(args)
        if ARGS_SPLIT_TOKEN in args.search:
            search_name, params = args.search.split(ARGS_SPLIT_TOKEN)
            if search_name not in SEARCH_METHOD_CLASS_NAMES:
                raise ValueError(f"Error: unsupported search {search_name}")
            search_method = eval(f"{SEARCH_METHOD_CLASS_NAMES[search_name]}({params})")
        elif args.search in SEARCH_METHOD_CLASS_NAMES:
            search_method = eval(f"{SEARCH_METHOD_CLASS_NAMES[args.search]}()")
        else:
            raise ValueError(f"Error: unsupported attack {args.search}")
    return textattack.shared.Attack(
        goal_function,
        constraints,
        transformation,
        search_method,
        constraint_cache_size=args.constraint_cache_size,
    )


def parse_logger_from_args(args):
    # Create logger
    attack_log_manager = textattack.loggers.AttackLogManager()

    # Get current time for file naming
    timestamp = time.strftime("%Y-%m-%d-%H-%M")

    # Get default directory to save results
    current_dir = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir, "outputs", "attacks"
    )
    out_dir_txt = out_dir_csv = os.path.normpath(outputs_dir)

    # Get default txt and csv file names
    if args.recipe:
        filename_txt = f"{args.model}_{args.recipe}_{timestamp}.txt"
        filename_csv = f"{args.model}_{args.recipe}_{timestamp}.csv"
    else:
        filename_txt = f"{args.model}_{timestamp}.txt"
        filename_csv = f"{args.model}_{timestamp}.csv"

    # if '--log-to-txt' specified with arguments
    if args.log_to_txt:
        # if user decide to save to a specific directory
        if args.log_to_txt[-1] == "/":
            out_dir_txt = args.log_to_txt
        # else if path + filename is given
        elif args.log_to_txt[-4:] == ".txt":
            out_dir_txt = args.log_to_txt.rsplit("/", 1)[0]
            filename_txt = args.log_to_txt.rsplit("/", 1)[-1]
        # otherwise, customize filename
        else:
            filename_txt = f"{args.log_to_txt}.txt"

    # if "--log-to-csv" is called
    if args.log_to_csv:
        # if user decide to save to a specific directory
        if args.log_to_csv[-1] == "/":
            out_dir_csv = args.log_to_csv
        # else if path + filename is given
        elif args.log_to_csv[-4:] == ".csv":
            out_dir_csv = args.log_to_csv.rsplit("/", 1)[0]
            filename_csv = args.log_to_csv.rsplit("/", 1)[-1]
        # otherwise, customize filename
        else:
            filename_csv = f"{args.log_to_csv}.csv"

    # in case directory doesn't exist
    if not os.path.exists(out_dir_txt):
        os.makedirs(out_dir_txt)
    if not os.path.exists(out_dir_csv):
        os.makedirs(out_dir_csv)

    # if "--log-to-txt" specified in terminal command (with or without arg), save to a txt file
    if args.log_to_txt == "" or args.log_to_txt:
        attack_log_manager.add_output_file(os.path.join(out_dir_txt, filename_txt))

    # if "--log-to-csv" specified in terminal command(with  or without arg), save to a csv file
    if args.log_to_csv == "" or args.log_to_csv:
        # "--csv-style used to swtich from 'fancy' to 'plain'
        color_method = None if args.csv_style == "plain" else "file"
        csv_path = os.path.join(out_dir_csv, filename_csv)
        attack_log_manager.add_output_csv(csv_path, color_method)
        textattack.shared.logger.info(f"Logging to CSV at path {csv_path}.")

    # Visdom
    if args.enable_visdom:
        attack_log_manager.enable_visdom()

    # Weights & Biases
    if args.enable_wandb:
        attack_log_manager.enable_wandb()

    # Stdout
    if not args.disable_stdout:
        attack_log_manager.enable_stdout()
    return attack_log_manager


def parse_checkpoint_from_args(args):
    file_name = os.path.basename(args.checkpoint_file)
    if file_name.lower() == "latest":
        dir_path = os.path.dirname(args.checkpoint_file)
        chkpt_file_names = [f for f in os.listdir(dir_path) if f.endswith(".ta.chkpt")]
        assert chkpt_file_names, "Checkpoint directory is empty"
        timestamps = [int(f.replace(".ta.chkpt", "")) for f in chkpt_file_names]
        latest_file = str(max(timestamps)) + ".ta.chkpt"
        checkpoint_path = os.path.join(dir_path, latest_file)
    else:
        checkpoint_path = args.checkpoint_file

    checkpoint = textattack.shared.Checkpoint.load(checkpoint_path)

    return checkpoint


def default_checkpoint_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(
        current_dir, os.pardir, os.pardir, os.pardir, "checkpoints"
    )
    return os.path.normpath(checkpoints_dir)


def merge_checkpoint_args(saved_args, cmdline_args):
    """Merge previously saved arguments for checkpoint and newly entered
    arguments."""
    args = copy.deepcopy(saved_args)
    # Newly entered arguments take precedence
    args.parallel = cmdline_args.parallel
    # If set, replace
    if cmdline_args.checkpoint_dir:
        args.checkpoint_dir = cmdline_args.checkpoint_dir
    if cmdline_args.checkpoint_interval:
        args.checkpoint_interval = cmdline_args.checkpoint_interval

    return args
