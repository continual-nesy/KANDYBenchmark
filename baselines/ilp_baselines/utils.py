import datetime
import random
import argparse
import yaml
import pandas as pd
import pyswip
import re

class ArgNumber:
    """Implement the notion of 'number' when passed as input argument to the program, deeply checking it.
    It also checks if the number falls in a given range."""

    def __init__(self, number_type: type(int) | type(float),
                 min_val: int | float | None = None,
                 max_val: int | float | None = None):
        self.__number_type = number_type
        self.__min = min_val
        self.__max = max_val
        if number_type not in [int, float]:
            raise argparse.ArgumentTypeError("Invalid number type (it must be int or float)")
        if not ((self.__min is None and self.__max is None) or
                (self.__max is None) or (self.__min is not None and self.__min < self.__max)):
            raise argparse.ArgumentTypeError("Invalid range")

    def __call__(self, value: int | float | str) -> int | float:
        try:
            val = self.__number_type(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value specified, conversion issues! Provided: {value}")
        if self.__min is not None and val < self.__min:
            raise argparse.ArgumentTypeError(f"Invalid value specified, it must be >= {self.__min}")
        if self.__max is not None and val > self.__max:
            raise argparse.ArgumentTypeError(f"Invalid value specified, it must be <= {self.__max}")
        return val


class ArgBoolean:
    """Implement the notion of 'boolean' when passed as input argument to the program, deeply checking it.
    It allows the user to provide it in different forms {1, 'True', 'true', 'yes'}, {0, 'False', 'false', 'no'}."""

    def __call__(self, value: str | bool | int) -> bool:
        if isinstance(value, str):
            val = value.lower().strip()
            if val != "true" and val != "false" and val != "yes" and val != "no":
                raise argparse.ArgumentTypeError(f"Invalid value specified: {value}")
            val = True if (val == "true" or val == "yes") else False
        elif isinstance(value, int):
            if value != 0 and value != 1:
                raise argparse.ArgumentTypeError(f"Invalid value specified: {value}")
            val = value == 1
        elif isinstance(value, bool):
            val = value
        else:
            raise argparse.ArgumentTypeError(f"Invalid value specified (expected boolean): {value}")
        return val

class ArgString:
    """Implement the notion of 'string' when passed as input argument to the program, deeply checking it.
    It allows the user to provide it in different forms {1, 'True', 'true', 'yes'}, {0, 'False', 'false', 'no'}."""

    def __call__(self, value: str | bool | int) -> bool:
        val = str(value)

        if val == "None":
            val = None
        return val

def generate_experiment_name(prefix: str | None = None, suffix: str | None = None) -> str:
    """Generate a dummy name for an experiment, such as 2023-08-10_17-06-22_Chianina.

        :param prefix: An optional prefix to add to the experiment name.
        :param suffix: An optional suffix to add to the experiment name (by default, it is a cattle breed).
        :returns: A string that represents the name of an experiment, such as 2023-08-10_17-06-22_Chianina.
    """

    def_suffix = ["Agerolese", "Angus", "Bianca_Modenese", "Bruna", "Burlina", "Cinisara", "Chianina", "Frisona",
                  "Grigio_Alpina", "Herens", "Limousine", "Marchigiana", "Maremmana", "Modicana", "Pezzata_Rossa",
                  "Piemontese", "Podolica", "Romagnola", "Simmental", "Valdostana", "Wagyu"]
    ret = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if prefix is not None:
        ret = prefix + "_" + ret
    if suffix is not None:
        ret = ret + "_" + suffix
    else:
        ret = ret + "_" + random.choice(def_suffix)
    return ret

def elapsed_time(from_seconds: float, to_seconds: float) -> str:
    """Format a string message with the time elapsed between two checkpoints.

        :param from_seconds: The float time of the first checkpoint.
        :param to_seconds: The float time of the second checkpoint.
        :returns: A message like this one: "3 hours, 24 minutes, 2.311 seconds".
    """

    elapsed = to_seconds - from_seconds
    minutes = int(elapsed / 60.)
    hours = int(elapsed / 60. / 60.)
    seconds = elapsed - hours * 60. * 60. - minutes * 60.
    return str(hours) + " hours, " + str(minutes) + " minutes, " + f"{seconds:.3f} seconds"



def symbol_to_prolog_natural(symbol: dict) -> str:
    if "shape" in symbol:
        return "{}_{}_{}".format(symbol["shape"], symbol["color"], symbol["size"])
    else:
        op = list(symbol.keys())[0]
        return "{}([{}])".format(op, ", ".join([symbol_to_prolog_natural(x) for x in symbol[op]]))


def build_problem_aleph_natural(df: pd.DataFrame, target_task_id: int) -> (str, str):
    pos_examples = []
    neg_examples = []
    examples = [] # Popper-like encoding to use a single compute_metrics function.

    for i in range(len(df)):
        x = df.iloc[i]
        if x["task_id"] == target_task_id:
            sample = symbol_to_prolog_natural(yaml.safe_load(x["symbol"]))
            if x["label"]:
                pos_examples.append("valid({}).".format(sample))
                examples.append("pos(valid({})).".format(sample))
            else:
                neg_examples.append("valid({}).".format(sample))
                examples.append("neg(valid({})).".format(sample))

    return "\n".join(sorted(list(pos_examples))), "\n".join(sorted(list(neg_examples))), "\n".join(sorted(list(examples)))


def build_problem_popper_natural(df: pd.DataFrame, target_task_id: int) -> str:
    examples = []

    for i in range(len(df)):
        x = df.iloc[i]
        if x["task_id"] == target_task_id:
            sample = symbol_to_prolog_natural(yaml.safe_load(x["symbol"]))
            examples.append("{}(valid({})).".format(("pos" if x["label"] else "neg"), sample))

    return "\n".join(sorted(list(examples)))


def symbol_to_prolog_pointer(symbol: dict, known_compositions: dict, known_atoms: set) -> (str, list, dict, set):
    out = []

    if str(symbol) not in known_compositions:
        known_compositions[str(symbol)] = "c{:06d}".format(len(known_compositions))

    assert len(symbol.keys()) == 1

    operator = list(symbol.keys())[0]
    children = []

    for c in symbol[operator]:
        if "shape" in c:
            children.append("{}_{}_{}".format(c["shape"], c["color"], c["size"]))
            known_atoms.add("{}_{}_{}".format(c["shape"], c["color"], c["size"]))
        else:
            child_comp, child_list, known_compositions, known_atoms = symbol_to_prolog_pointer(c, known_compositions,
                                                                                               known_atoms)
            out.extend(child_list)
            children.append(child_comp)

    out.insert(0, (known_compositions[str(symbol)], operator, children))

    return known_compositions[str(symbol)], out, known_compositions, known_atoms


def build_problem_aleph_pointer(df: pd.DataFrame, task_id: int) -> (str, str, str):
    known_compositions = {}
    known_atoms = set()

    bg = set()
    pos_examples = set()
    neg_examples = set()
    examples = set()  # Popper-like encoding to use a single compute_metrics function.

    for i in range(len(df)):
        if df.iloc[i]["task_id"] == task_id:
            x = df.iloc[i]

            sample_name = "t{:03d}_s{:06d}".format(df.iloc[i]["task_id"], len(pos_examples) + len(neg_examples))

            if str(x["symbol"]) not in known_compositions:
                known_compositions[str(x["symbol"])] = "c{:06d}".format(len(known_compositions))

            bg.add("sample_is({}, {}).".format(sample_name, known_compositions[str(x["symbol"])]))

            _, children, known_compositions, known_atoms = symbol_to_prolog_pointer(yaml.safe_load(x["symbol"]),
                                                                                known_compositions, known_atoms)

            for comp in children:
                bg.add("defined_as({}, {}, [{}]).".format(comp[0], comp[1],
                                                                  ", ".join([str(c) for c in comp[2]])))
            if x["label"]:
                pos_examples.add("valid({}).".format(sample_name))
                examples.add("pos(valid({})).".format(sample_name))
            else:
                neg_examples.add("valid({}).".format(sample_name))
                examples.add("neg(valid({})).".format(sample_name))

    for a in known_atoms:
        bg.add("atomic_obj({}).".format(a))

    return "\n".join(sorted(list(bg))), "\n".join(sorted(list(pos_examples))), "\n".join(sorted(list(neg_examples))), "\n".join(sorted(list(examples)))

def build_problem_popper_pointer(df: pd.DataFrame, task_id: int) -> (str, str):
    known_compositions = {}
    known_atoms = set()

    bg = set()
    examples = set()

    for i in range(len(df)):
        if df.iloc[i]["task_id"] == task_id:
            x = df.iloc[i]

            sample_name = "t{:03d}_s{:06d}".format(df.iloc[i]["task_id"], len(examples))

            if str(x["symbol"]) not in known_compositions:
                known_compositions[str(x["symbol"])] = "c{:06d}".format(len(known_compositions))

            bg.add("sample_is({}, {}).".format(sample_name, known_compositions[str(x["symbol"])]))

            _, children, known_compositions, known_atoms = symbol_to_prolog_pointer(yaml.safe_load(x["symbol"]),
                                                                                known_compositions, known_atoms)

            for comp in children:
                bg.add("defined_as({}, {}, [{}]).".format(comp[0], comp[1],
                                                                  ", ".join([str(c) for c in comp[2]])))
            examples.add("{}(valid({})).".format(("pos" if x["label"] else "neg"), sample_name))

    for a in known_atoms:
        bg.add("atomic_obj({}).".format(a))

    return "\n".join(sorted(list(bg))), "\n".join(sorted(list(examples)))

def compute_metrics(bg_file: str, theory: dict, examples: str) -> dict:
    metrics = {}
    pl = pyswip.Prolog()

    pl.assertz("tp(N) :- aggregate_all(count, (pos(Y), arg(1, Y, X), distinct(valid(X))), N)")
    pl.assertz("tn(N) :- aggregate_all(count, (neg(Y), arg(1, Y, X), distinct(not(valid(X)))), N)")
    pl.assertz("fp(N) :- aggregate_all(count, (neg(Y), arg(1, Y, X), distinct(valid(X))), N)")
    pl.assertz("fn(N) :- aggregate_all(count, (pos(Y), arg(1, Y, X), distinct(not(valid(X)))), N)")

    for task_id in theory.keys():
        next(pl.query("dynamic(pos/1)"))
        next(pl.query("dynamic(neg/1)"))

        if theory[task_id] is None:
            metrics[task_id] = {"acc": 0.0, "pr": 0.0, "rec": 0.0, "f1": 0.0, "macro_acc": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
        else:
            with open(bg_file.format(task_id), "r") as file:
                for r in file.read().split("\n"):
                    if re.match("\w*[a-z]+.*\.$", r):
                        pl.assertz(r.split(".")[0]) # We assert the file line by line, since we cannot retract it later with consult...

            with open(examples.format(task_id)) as file:
                for e in file.read().split("\n"):
                    if re.match("\w*[a-z]+.*\.$", e):
                        pl.assertz(e.split(".")[0])

            for p in theory[task_id].split("."):
                if re.match("\w*[a-z]+.*", p):
                    pl.assertz(p) # TODO: SE i predicati sono multipli fare split su \n?

            out = list(pl.query("tp(TP), tn(TN), fp(FP), fn(FN)", maxresult=1))

            acc = 0.0
            pr = 0.0
            rec = 0.0
            f1 = 0.0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            macro_acc = 0.0

            if len(out) > 0:
                tp = float(out[0]["TP"])
                tn = float(out[0]["TN"])
                fp = float(out[0]["FP"])
                fn = float(out[0]["FN"])

                if tp + tn + fp + fn > 0:
                    acc = (tp + tn) / (tp + tn + fp + fn)
                if tp + fp > 0:
                    pr = tp / (tp + fp)
                if tp + fn > 0:
                    rec = tp / (tp + fn)
                if tn + fp > 0:
                    fpr = tn / (tn + fp)
                if pr + rec > 0:
                    f1 = 2 * pr * rec / (pr + rec)

                macro_acc = (rec + fpr) / 2


            metrics[task_id] = {"acc": acc, "pr": pr, "rec": rec, "f1": f1, "macro_acc": macro_acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


            with open(bg_file.format(task_id), "r") as file:
                for r in file.read().split("\n"):
                    if re.match("\w*[a-z]+.*\.$", r):
                        head = r.split(":-")[0]
                        pred = head.split("(")[0].strip()
                        for arity in range(10):  # Ugly hack to abolish all possible arities...
                            next(pl.query("abolish({}/{})".format(pred, arity)))

            for p in theory[task_id].split("."):
                if re.match("\w*[a-z]+.*", p):
                    head = p.split(":-")[0]
                    pred = head.split("(")[0].strip()
                    for arity in range(10):  # Ugly hack to abolish all possible arities...
                        next(pl.query("abolish({}/{})".format(pred, arity)))

            next(pl.query("abolish(pos/1)"))
            next(pl.query("abolish(neg/1)"))


    return metrics