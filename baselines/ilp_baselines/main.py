import argparse
import os
import sys
import time
import wandb
import tempfile
import pandas as pd
import yaml



from utils import *
from run_popper import prepare_files_popper, run_popper
from run_aleph import prepare_files_aleph, run_aleph



# initial checks
assert __name__ == "__main__", "Invalid usage! Run this script from command line, do not import it!"
if len(sys.argv) == 1:
    print("Not enough arguments were provided.\nRun with -h to get the list of supported arguments.")
    sys.exit(0)

# Customizable options
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--output_folder", help="Output folder (default: exp)", type=ArgString(), default="exp")
arg_parser.add_argument("--save_options", help="Save options at the beginning of experiment.",
                        type=ArgBoolean(), default=True)
arg_parser.add_argument("--wandb_project", help="Use W&B, this is the project name (default: None)",
                        type=ArgString(), default=None)
arg_parser.add_argument("--wandb_group", help="Group within the W&B project name (default: None)",
                        type=ArgString(), default=None)
arg_parser.add_argument('--csv_dir', help="Annotations folder (default .)", type=ArgString(), default=".")
arg_parser.add_argument('--train_csv', help="Train annotations", type=ArgString(), default="train_annotations.csv")
arg_parser.add_argument('--val_csv', help="Validation annotations (optional)", type=ArgString(), default=None)
arg_parser.add_argument('--test_csv', help="Test annotations (optional)", type=ArgString(), default=None)
arg_parser.add_argument('--prefix', help="Background knowledge and bias prefix (looking for _{natural|pointer}_bg.pl and _{natural|pointer}_{popper|aleph}_bias.pl files)", type=ArgString(), default="minimal")
arg_parser.add_argument('--prefix_cheat', help="Cheat predicates prefix (looking for _{natural|pointer}_bg.pl and _{natural|pointer}_{popper|aleph}_bias.pl files)", type=ArgString(), default=None)
arg_parser.add_argument('--engine', help="The engine to be used in "
                                         "{'aleph', 'popper'}", type=ArgString(), default="popper", choices=["aleph", "popper"])
arg_parser.add_argument('--encoding', help="The encoding to be used in "
                                         "{'natural', 'pointer'}", type=ArgString(), default="natural", choices=["natural", "pointer"])
arg_parser.add_argument('--timeout', help="Timeout for execution (in seconds)", type=ArgNumber(int, min_val=0, max_val=86400), default=600)

# Shared biases:
arg_parser.add_argument('--max_vars', help="Maximum number of variables in body", type=ArgNumber(int, min_val=0), default=6)
arg_parser.add_argument('--max_clauses', help="Maximum number of disjunctive clauses in the theory", type=ArgNumber(int, min_val=1), default=1)
arg_parser.add_argument('--max_size', help="Maximum number of literals in the body of a clause", type=ArgNumber(int, min_val=1), default=6)

# Popper-specific biases:
arg_parser.add_argument('--singleton_vars', help="Enable singleton variables (Popper only)", type=ArgBoolean(), default=False)
arg_parser.add_argument('--predicate_invention', help="Enable predicate invention (Popper only)", type=ArgBoolean(), default=False)
arg_parser.add_argument('--recursion', help="Enable recursion (Popper only)", type=ArgBoolean(), default=False)
arg_parser.add_argument('--max_literals', help="Maximum number of literals in the entire theory (Popper only)", type=ArgNumber(int, min_val=1), default=40)

# Aleph-specific biases:
arg_parser.add_argument('--min_acc', help="Minimum acceptable accuracy (aleph only)", type=ArgNumber(float, min_val=0.0, max_val=1.0), default=0.5)
arg_parser.add_argument('--noise', help="Maximum acceptable number of false positives (aleph only)", type=ArgNumber(int, min_val=0), default=0)
arg_parser.add_argument('--i', help="Layers on new variables (aleph only)", type=ArgNumber(int, min_val=1), default=2)
arg_parser.add_argument('--min_pos', help="Minimum number of positives covered by a clause (aleph only)", type=ArgNumber(int, min_val=1), default=2)
arg_parser.add_argument('--depth', help="Proof depth (aleph only)", type=ArgNumber(int, min_val=1), default=10)
arg_parser.add_argument('--nodes', help="nodes to explore during search (aleph only)", type=ArgNumber(int, min_val=1), default=5000)


opts = vars(arg_parser.parse_args())

# setup W&B
wb = None
if opts['wandb_project'] is not None:
    if opts['wandb_group'] is not None:
        wb = wandb.init(project=opts['wandb_project'], group=opts['wandb_group'], config=opts)
    else:
        wb = wandb.init(project=opts['wandb_project'], config=opts)

# Reject illegal combinations.
assert opts["prefix"] != "minimal" or opts["prefix_cheat"] is None, "Cannot use cheat predicates together with the minimal knowledge base."
assert opts["prefix_cheat"] is None or (not opts["predicate_invention"] and not opts["recursion"] and not opts["singleton_vars"]), "Cannot use predicate invention, recursion or singletons with cheat knowledge base."
assert opts["engine"] != "aleph" or (not opts["predicate_invention"] and not opts["recursion"] and not opts["singleton_vars"]), "Cannot use predicate invention, recursion or singletons with engine aleph."


train_df = pd.read_csv(os.path.join(opts["csv_dir"], opts["train_csv"]))
val_df = pd.read_csv(os.path.join(opts["csv_dir"], opts["val_csv"])) if opts["val_csv"] is not None else None
test_df = pd.read_csv(os.path.join(opts["csv_dir"], opts["test_csv"])) if opts["test_csv"] is not None else None

start_time = time.time()

with tempfile.TemporaryDirectory() as tmp_dir:
    if opts["engine"] == "popper":
        prepare_files_popper(tmp_dir, opts, train_df, val_df, test_df)
        progs = run_popper(train_df, tmp_dir, opts)

    elif opts["engine"] == "aleph":
        prepare_files_aleph(tmp_dir, opts, train_df, val_df, test_df)
        progs = run_aleph(train_df, tmp_dir, opts)

    if opts["encoding"] == "natural":
        train_bg = os.path.join(tmp_dir, "bg.pl")
    else:
        train_bg = os.path.join(tmp_dir, "bg_{}.pl")  # Format() is inside compute metrics.
    train_prefix = os.path.join(tmp_dir, "{}.pl")

    train_metrics = compute_metrics(train_bg, progs, train_prefix)

    if opts["encoding"] == "natural":
        test_bg = os.path.join(tmp_dir, "bg_test.pl")
        val_bg = os.path.join(tmp_dir, "bg_val.pl")
    else:
        test_bg = os.path.join(tmp_dir, "bg_{}_test.pl")
        val_bg = os.path.join(tmp_dir, "bg_{}_val.pl")
    test_prefix = os.path.join(tmp_dir, "{}_test.pl")
    test_metrics = compute_metrics(test_bg, progs, test_prefix)
    val_prefix = os.path.join(tmp_dir, "{}_val.pl")
    val_metrics = compute_metrics(val_bg, progs, val_prefix)

results = {}

results["train"] = [({k: v for k, v in train_metrics[tid].items()} if tid in train_metrics
                     else {k: 0 for k in ["acc", "pr", "rec", "f1", "macro_acc", "tp", "tn", "fp", "fn"]})
                    for tid in range(max(train_metrics.keys()) + 1)]
results["val"] = [({k: v for k, v in val_metrics[tid].items()} if tid in val_metrics
                     else {k: 0 for k in ["acc", "pr", "rec", "f1", "macro_acc", "tp", "tn", "fp", "fn"]})
                    for tid in range(max(val_metrics.keys()) + 1)]
results["test"] = [({k: v for k, v in test_metrics[tid].items()} if tid in test_metrics
                     else {k: 0 for k in ["acc", "pr", "rec", "f1", "macro_acc", "tp", "tn", "fp", "fn"]})
                    for tid in range(max(test_metrics.keys()) + 1)]



opts["time"] = elapsed_time(start_time, time.time())

exp_name = generate_experiment_name()
output_folder = os.path.abspath(opts['output_folder'])
opts['exp_name'] = exp_name
opts['command_line'] = "python " + (" ".join("\""+arg+"\"" if " " in arg else arg for arg in sys.argv))

tmp = {"opts": {k: v for k, v in opts.items()},
       "theories": [(progs[tid] if tid in progs else None) for tid in range(max(progs.keys()) + 1)],
       "results": {k: v for k, v in results.items()}}

tmp["avg_test_macro_acc"] = sum([x["macro_acc"] for x in results["test"]]) / len(results["test"])
tmp["avg_val_macro_acc"] = sum([x["macro_acc"] for x in results["val"]]) / len(results["val"])


if opts['save_options']:
    os.makedirs(output_folder, exist_ok=True)
    out_file = os.path.join(output_folder, "{}.yml".format(opts['exp_name']))
    with open(out_file, "w") as file:
        yaml.dump(tmp, file, default_flow_style=False)


# logging to W&B
if opts['wandb_project'] is not None:
    wb.log(tmp)

    wb.finish()
