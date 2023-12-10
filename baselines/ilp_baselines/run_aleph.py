import typing
import multiprocessing
from multiprocessing import Process


from utils import build_problem_aleph_natural, build_problem_aleph_pointer
import pandas as pd
import os

from PyILP import PyILP

def prepare_files_aleph(path: str, config: dict, train_df: pd.DataFrame, val_df: typing.Optional[pd.DataFrame] = None, test_df: typing.Optional[pd.DataFrame] = None) -> None:
    assert config["encoding"] in ["natural", "pointer"], "Invalid encoding."
    for task_id in train_df["task_id"].unique():
        ex_file = os.path.join(path, "{}.pl".format(task_id))
        p_ex_file = os.path.join(path, "{}.f".format(task_id))
        n_ex_file = os.path.join(path, "{}.n".format(task_id))
        if config["encoding"] == "natural":
            pos_examples, neg_examples, examples = build_problem_aleph_natural(train_df, task_id)
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
            bg, pos_examples, neg_examples, examples = build_problem_aleph_pointer(train_df, task_id)
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
        with open(p_ex_file, "w") as file:
            file.write(pos_examples)
        with open(n_ex_file, "w") as file:
            file.write(neg_examples)

    if val_df is not None:
        for task_id in val_df["task_id"].unique():
            ex_file = os.path.join(path, "{}_val.pl".format(task_id))
            p_ex_file = os.path.join(path, "{}_val.f".format(task_id))
            n_ex_file = os.path.join(path, "{}_val.n".format(task_id))
            if config["encoding"] == "natural":
                pos_examples, neg_examples, examples = build_problem_aleph_natural(val_df, task_id)
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
                bg, pos_examples, neg_examples, examples = build_problem_aleph_pointer(val_df, task_id)
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
            with open(p_ex_file, "w") as file:
                file.write(pos_examples)
            with open(n_ex_file, "w") as file:
                file.write(neg_examples)

    if test_df is not None:
        for task_id in test_df["task_id"].unique():
            ex_file = os.path.join(path, "{}_test.pl".format(task_id))
            p_ex_file = os.path.join(path, "{}_test.f".format(task_id))
            n_ex_file = os.path.join(path, "{}_test.n".format(task_id))
            if config["encoding"] == "natural":
                pos_examples, neg_examples, examples = build_problem_aleph_natural(test_df, task_id)
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
                bg, pos_examples, neg_examples, examples = build_problem_aleph_pointer(test_df, task_id)
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
            with open(p_ex_file, "w") as file:
                file.write(pos_examples)
            with open(n_ex_file, "w") as file:
                file.write(neg_examples)

    bias_file = os.path.join(path, "bias.pl")
    with open(bias_file, "w") as file:
        file.write(":- aleph_set(verbosity, 0).\n")
        if "max_vars" in config:
            file.write(":- aleph_set(newvars, {}).\n".format(config["max_vars"]))
        if "max_clauses" in config:
            file.write(":- aleph_set(clauses, {}).\n".format(config["max_clauses"]))
        if "max_size" in config:
            file.write(":- aleph_set(clauselength, {}).\n".format(config["max_size"]))
        if "min_acc" in config:
            file.write(":- aleph_set(min_acc, {}).\n".format(config["min_acc"]))
        if "noise" in config:
            file.write(":- aleph_set(noise, {}).\n".format(config["noise"]))
        if "i" in config:
            file.write(":- aleph_set(i, {}).\n".format(config["i"]))
        if "min_pos" in config:
            file.write(":- aleph_set(min_pos, {}).\n".format(config["min_pos"]))
        if "depth" in config:
            file.write(":- aleph_set(depth, {}).\n".format(config["depth"]))
        if "nodes" in config:
            file.write(":- aleph_set(nodes, {}).\n".format(config["nodes"]))
        if "timeout" in config:
            file.write(":- aleph_set(timeout, {}).\n".format(config["timeout"]))


        with open("{}_{}_aleph_bias.pl".format(config["prefix"], config["encoding"]), "r") as file2:
            file.write(file2.read())
        if config["prefix_cheat"] is not None:
            file.write("\n")
            with open("{}_{}_aleph_bias.pl".format(config["prefix_cheat"], config["encoding"]), "r") as file2:
                file.write(file2.read())

def run_aleph(df: pd.DataFrame, path: str, config: dict) -> dict:
    def exec(bg, pos, neg, bias, values):
        model = PyILP.aleph_learn(file=bg, positive_example=pos, negative_example=neg, test_size=0.0,  settings=bias)
        if model is not None:
            values["prog"] = "\n".join(model.hypothesis)
        else:
            values["prog"] = None


    out = {}

    bias = os.path.join(path, "bias.pl")

    if "timeout" in config:
        timeout = config["timeout"]
    else:
        timeout = None


    for task_id in df["task_id"].unique():
        if df.groupby("task_id")["label"].sum()[task_id] > 0:
            if config["encoding"] == "natural":
                bg = os.path.join(path, "bg.pl")
            else:
                bg = os.path.join(path, "bg_{}.pl".format(task_id))

            pos = os.path.join(path, "{}.f".format(task_id))
            neg = os.path.join(path, "{}.n".format(task_id))

            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = Process(target=exec, args=(bg, pos, neg, bias, return_dict))
            p.start()
            if timeout is None:
                p.join()
            else:
                p.join(timeout)
                if p.exitcode is None:
                    p.terminate()
                    p.join()

            if "prog" in return_dict and return_dict["prog"] is not None:
                out[task_id] = return_dict["prog"]
            else:
                out[task_id] = None

        else:
            print("Task {} has no positive examples to learn from.".format(task_id))
            out[task_id] = None

    return out