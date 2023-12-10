import typing
import multiprocessing
from multiprocessing import Process

from popper.util import Settings, format_prog, order_prog
from popper.loop import learn_solution

from utils import build_problem_popper_natural, build_problem_popper_pointer
import pandas as pd
import os

def prepare_files_popper(path: str, config: dict, train_df: pd.DataFrame, val_df: typing.Optional[pd.DataFrame] = None, test_df: typing.Optional[pd.DataFrame] = None) -> None:
    assert config["encoding"] in ["natural", "pointer"], "Invalid encoding."
    for task_id in train_df["task_id"].unique():
        ex_file = os.path.join(path, "{}.pl".format(task_id))
        if config["encoding"] == "natural":
            examples = build_problem_popper_natural(train_df, task_id)
            assert "prefix" in config, "Missing prefix."
            bg_file = os.path.join(path, "bg.pl")
            with open(bg_file, "w") as file:
                with open("{}_{}_bg.pl".format(config["prefix"], config["encoding"]), "r") as file2:
                    file.write(file2.read())
                if config["prefix_cheat"] is not None:
                    file.write("\n")
                    with open("{}_{}_bg.pl".format(config["prefix_cheat"], config["encoding"]), "r") as file2:
                        file.write(file2.read())
        else:
            bg, examples = build_problem_popper_pointer(train_df, task_id)
            assert "prefix" in config, "Missing prefix."
            bg_file = os.path.join(path, "bg_{}.pl".format(task_id))
            with open(bg_file, "w") as file:
                with open("{}_{}_bg.pl".format(config["prefix"], config["encoding"]), "r") as file2:
                    file.write(file2.read())
                if config["prefix_cheat"] is not None:
                    file.write("\n")
                    with open("{}_{}_bg.pl".format(config["prefix_cheat"], config["encoding"]), "r") as file2:
                        file.write(file2.read())
                file.write("\n")
                file.write(bg)

        with open(ex_file, "w") as file:
            file.write(examples)

    if val_df is not None:
        for task_id in val_df["task_id"].unique():
            ex_file = os.path.join(path, "{}_val.pl".format(task_id))
            if config["encoding"] == "natural":
                examples = build_problem_popper_natural(val_df, task_id)
                assert "prefix" in config, "Missing prefix."
                bg_file = os.path.join(path, "bg_val.pl")
                with open(bg_file, "w") as file:
                    with open("{}_{}_bg.pl".format(config["prefix"], config["encoding"]), "r") as file2:
                        file.write(file2.read())
                    if config["prefix_cheat"] is not None:
                        file.write("\n")
                        with open("{}_{}_bg.pl".format(config["prefix_cheat"], config["encoding"]),
                                  "r") as file2:
                            file.write(file2.read())
            else:
                bg, examples = build_problem_popper_pointer(val_df, task_id)
                assert "prefix" in config, "Missing prefix."
                bg_file = os.path.join(path, "bg_{}_val.pl".format(task_id))
                with open(bg_file, "w") as file:
                    with open("{}_{}_bg.pl".format(config["prefix"], config["encoding"]), "r") as file2:
                        file.write(file2.read())
                    if config["prefix_cheat"] is not None:
                        file.write("\n")
                        with open("{}_{}_bg.pl".format(config["prefix_cheat"], config["encoding"]),
                                  "r") as file2:
                            file.write(file2.read())
                    file.write("\n")
                    file.write(bg)
            with open(ex_file, "w") as file:
                file.write(examples)

    if test_df is not None:
        for task_id in test_df["task_id"].unique():
            ex_file = os.path.join(path, "{}_test.pl".format(task_id))
            if config["encoding"] == "natural":
                examples = build_problem_popper_natural(test_df, task_id)
                assert "prefix" in config, "Missing prefix."
                bg_file = os.path.join(path, "bg_test.pl")
                with open(bg_file, "w") as file:
                    with open("{}_{}_bg.pl".format(config["prefix"], config["encoding"]), "r") as file2:
                        file.write(file2.read())
                    if config["prefix_cheat"] is not None:
                        file.write("\n")
                        with open("{}_{}_bg.pl".format(config["prefix_cheat"], config["encoding"]),
                                  "r") as file2:
                            file.write(file2.read())
            else:
                bg, examples = build_problem_popper_pointer(test_df, task_id)
                assert "prefix" in config, "Missing prefix."
                bg_file = os.path.join(path, "bg_{}_test.pl".format(task_id))
                with open(bg_file, "w") as file:
                    with open("{}_{}_bg.pl".format(config["prefix"], config["encoding"]), "r") as file2:
                        file.write(file2.read())
                    if config["prefix_cheat"] is not None:
                        file.write("\n")
                        with open("{}_{}_bg.pl".format(config["prefix_cheat"], config["encoding"]),
                                  "r") as file2:
                            file.write(file2.read())
                    file.write("\n")
                    file.write(bg)
            with open(ex_file, "w") as file:
                file.write(examples)

    bias_file = os.path.join(path, "bias.pl")
    with open(bias_file, "w") as file:
        if "singleton_vars" in config and config["singleton_vars"]:
            file.write("allow_singletons.\n")
        if "predicate_invention" in config and config["predicate_invention"]:
            file.write("enable_pi.\n")
        if "recursion" in config and config["recursion"]:
            file.write("enable_recursion.\n")
        if "max_vars" in config:
            file.write("max_vars({}).\n".format(config["max_vars"]))
        if "max_clauses" in config:
            file.write("max_clauses({}).\n".format(config["max_clauses"]))
        if "max_size" in config:
            file.write("max_body({}).\n".format(config["max_size"]))

        with open("{}_{}_popper_bias.pl".format(config["prefix"], config["encoding"]), "r") as file2:
            file.write(file2.read())
        if config["prefix_cheat"] is not None:
            file.write("\n")
            with open("{}_{}_popper_bias.pl".format(config["prefix_cheat"], config["encoding"]), "r") as file2:
                file.write(file2.read())

def run_popper(df: pd.DataFrame, path: str, config: dict) -> dict:
    def exec(settings, values):
        prog, _, _ = learn_solution(settings)
        values['prog'] = prog

    if "timeout" in config:
        timeout = config["timeout"]
    else:
        timeout = None

    out = {}

    bias_file = os.path.join(path, "bias.pl")

    if "max_literals" in config:
        max_literals = config["max_literals"]
    else:
        max_literals = None

    for task_id in df["task_id"].unique():
        if df.groupby("task_id")["label"].sum()[task_id] > 0:
            if config["encoding"] == "natural":
                bg_file = os.path.join(path, "bg.pl")
            else:
                bg_file = os.path.join(path, "bg_{}.pl".format(task_id))
            settings = Settings(timeout=timeout, ex_file=os.path.join(path, "{}.pl".format(task_id)),
                                bk_file=bg_file, bias_file=bias_file, max_literals=max_literals)

            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = Process(target=exec, args=(settings, return_dict))
            p.start()
            if timeout is None:
                p.join()
            else:
                p.join(timeout)
                if p.exitcode is None:
                    p.terminate()
                    p.join()

            if "prog" in return_dict and return_dict["prog"] is not None:
                out[task_id] = format_prog(order_prog(return_dict["prog"]))

            else:
                out[task_id] = None

        else:
            print("Task {} has no positive examples to learn from.".format(task_id))
            out[task_id] = None

    return out

