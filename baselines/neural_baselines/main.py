import argparse
import os
import pprint
import sys
import time
import wandb
from os.path import join
from training import train
from dataset import check_data_folder, TaskOrganizedDataset
from utils import ArgNumber, ArgBoolean, save_dict, generate_experiment_name, set_seed, elapsed_time
from networks import generate_net, save_net

# initial checks
assert __name__ == "__main__", "Invalid usage! Run this script from command line, do not import it!"
if len(sys.argv) == 1:
    print("Not enough arguments were provided.\nRun with -h to get the list of supported arguments.")
    sys.exit(0)

# customizable options
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_path', help="Path to the root of the data folder", type=str, default="./")
arg_parser.add_argument("--model", help="The type of neural net to consider in "
                                        "{'mlp', 'cnn', 'resnet50', 'resnet50_head_only', 'vit_head_only'}",
                        type=str, default='mlp', choices=['mlp', 'cnn', 'resnet50',
                                                          'resnet50_head_only', 'vit_head_only'])
arg_parser.add_argument("--train", help="Training scheme in "
                                        "{'joint', 'independent', 'continual_task', 'continual_online'}",
                        type=str, default='joint',
                        choices=['joint', 'independent', 'continual_task', 'continual_online'])
arg_parser.add_argument("--supervised_only", help="Consider only supervised data (default: false)",
                        type=ArgBoolean(), default=False)
arg_parser.add_argument("--augment", help="Random augmentation of the training data (default: true)",
                        type=ArgBoolean(), default=True)
arg_parser.add_argument("--lr", help="Learning rate (negative -> Adam)", type=ArgNumber(float), default=-0.001)
arg_parser.add_argument("--weight_decay", help="Weight decay factor (default: 0.)",
                        type=ArgNumber(float, min_val=0.), default=0.)
arg_parser.add_argument("--batch", help="Mini-batch size (default: 16)", type=ArgNumber(int, min_val=1), default=16)
arg_parser.add_argument("--task_epochs",
                        help="Number of epochs over each task (incompatible with --train continual_online; default: 1)",
                        type=ArgNumber(int, min_val=1), default=1)
arg_parser.add_argument("--balance", help="Virtually augment the training data by repeating the positive or negative "
                                          "examples (per task) so that they will be the same number "
                                          "(incompatible with --train continual_online; default: false)",
                        type=ArgBoolean(), default=False)
arg_parser.add_argument("--replay_buffer",
                        help="Size of the experience replay buffer (only compatible with --train continual_*; "
                             "default: 0)",
                        type=ArgNumber(int, min_val=0), default=0)
arg_parser.add_argument("--replay_lambda",
                        help="Weight of the portion of the loss that is about experience replay (only compatible with"
                             " --train continual_*; default: 0.)",
                        type=ArgNumber(float, min_val=0.), default=0.)
arg_parser.add_argument("--seed", help="Integer seed for random numbers (if < 0, it depends on time, default case)",
                        type=int, default=-1)
arg_parser.add_argument("--output_folder", help="Output folder (default: exp)", type=str, default="exp")
arg_parser.add_argument("--device", help="Device to use (default: cpu, or the value of environment variable DEVICE, "
                                         "if available - set with 'export DEVICE=cuda:0', for example)",
                        type=str, default="cpu")
arg_parser.add_argument("--save_net", help="Save network at the end of experiment.", type=ArgBoolean(), default=True)
arg_parser.add_argument("--save_results", help="Save results at the end of experiment.",
                        type=ArgBoolean(), default=True)
arg_parser.add_argument("--save_options", help="Save options at the beginning of experiment.",
                        type=ArgBoolean(), default=True)
arg_parser.add_argument("--print_every", help="Number of gradient steps before consecutive prints to screen "
                                              "(default: 10)", type=int, default=10)
arg_parser.add_argument("--wandb_project", help="Use W&B, this is the project name (default: None)",
                        type=str, default=None)
arg_parser.add_argument("--wandb_group", help="Group within the W&B project name (default: None)",
                        type=str, default=None)
opts = vars(arg_parser.parse_args())

# checking data
print("Checking data folder (" + str(opts['data_path']) + ")...")
data_folder = os.path.abspath(opts['data_path'])
check_data_folder(data_folder)

# creating output folder and saving options to disk
exp_name = generate_experiment_name()
output_folder = os.path.abspath(opts['output_folder'])
opts['exp_name'] = exp_name
opts['command_line'] = "python " + (" ".join("\""+arg+"\"" if " " in arg else arg for arg in sys.argv))
if not os.path.exists(output_folder) and (opts['save_options'] or opts['save_net'] or opts['save_results']):
    os.makedirs(output_folder)
if opts['save_options']:
    save_dict(join(output_folder, opts['exp_name'] + "_options.json"), opts)
if os.environ.get('DEVICE') is not None:
    opts['device'] = os.environ.get('DEVICE')

# printing options to screen
print("*** Options ***")
pprint.pprint(opts, width=200)

# setup W&B
wb = None
if opts['wandb_project'] is not None:
    if opts['wandb_group'] is not None:
        wb = wandb.init(project=opts['wandb_project'], group=opts['wandb_group'], config=opts)
    else:
        wb = wandb.init(project=opts['wandb_project'], config=opts)

# checking dependencies among different options
assert opts['task_epochs'] == 1 or opts['train'] != 'continual_online', \
    "More than one task epoch (option: 'task_epochs') makes sense only if training (options: 'train') proceeds in " \
    "a way that is different from 'continual_online'."
assert (opts['train'] == 'continual_online' and opts['balance'] is True and opts['replay_buffer'] > 1) or \
       (opts['train'] == 'continual_online' and opts['balance'] is False) or \
       opts['train'] != 'continual_online', "Asking to 'balance' the training data makes sense only if training " \
                                            "(options: 'train') proceeds in a way that is different from " \
                                            "'continual_online' or, in case of 'continual_online, " \
                                            "if a replay buffer is used."
assert opts['replay_buffer'] == 0 or (opts['train'] == 'continual_task' or opts['train'] == 'continual_online'), \
    "You can only use a replay buffer when the training method ('train') is 'continual_task' or 'continual_online'"
assert opts['replay_buffer'] == 0 or opts['replay_lambda'] > 0., \
    "The 'replay_lambda' coefficient must be > 0., otherwise 'replay_buffer' will have no effects."
assert opts['replay_lambda'] == 0. or opts['replay_buffer'] > 0., \
    "The 'replay_buffer' must be > 0, otherwise 'replay_lambda' will have no effects."

# setting up seeds for random number generators
set_seed(opts['seed'])

# preparing data sets
print("Preparing datasets...")
train_set = TaskOrganizedDataset(join(data_folder, 'train'),
                                 supervised_only=opts['supervised_only'],
                                 max_buffer_size=opts['replay_buffer'])  # memory buffer for experience replay
val_set = TaskOrganizedDataset(join(data_folder, 'val'),
                               supervised_only=opts['supervised_only'])
test_set = TaskOrganizedDataset(join(data_folder, 'test'),
                                supervised_only=opts['supervised_only'])

# creating network
print("Creating network (" + opts['model'] + ")...")
if opts['train'] != 'independent':
    net, train_transforms, eval_transforms = generate_net(opts['model'],
                                                          num_outputs=train_set.num_tasks,
                                                          input_shape=train_set.input_shape)
else:
    net = []
    train_transforms = None
    eval_transforms = None
    for i in range(train_set.num_tasks):
        _net, train_transforms, eval_transforms = generate_net(opts['model'],
                                                               num_outputs=1,  # assuming binary tasks
                                                               input_shape=train_set.input_shape)
        net.append(_net)

# setting up transformations for data augmentation
train_set.transform = train_transforms if opts['augment'] else eval_transforms
val_set.transform = eval_transforms
test_set.transform = eval_transforms

# running experiment
start_time = time.time()
print("Running the training procedure (" + opts['train'] + ")...")
metrics_train, metrics_val, metrics_test = train(net, train_set, val_set, test_set, opts)

# logging to W&B
if wb is not None:
    print("Logging to W&B...")
    for i in range(0, train_set.num_tasks):
        for score_name in ['avg_accuracy', 'avg_forgetting', 'backward_transfer', 'forward_transfer']:
            wb.log(data={score_name + "-" + metrics_train['name']: metrics_train[score_name][i],
                         score_name + "-" + metrics_val['name']: metrics_val[score_name][i],
                         score_name + "-" + metrics_test['name']: metrics_test[score_name][i]},
                   step=i)

    columns = ["@"] + ["eval_" + str(i) for i in range(0, train_set.num_tasks)]
    score_name = 'acc_matrix'
    for metrics in [metrics_train, metrics_val, metrics_test]:
        scores = metrics[score_name]
        tab = wandb.Table(columns=columns, data=[["train_" + str(i)] + row for i, row in enumerate(scores)])
        wb.log({score_name + "-" + metrics['name']: tab})

    wb.finish()

# saving network and results
if opts['save_net']:
    print("Saving net...")
    save_net(net, join(output_folder, opts['exp_name'] + "_net.pth"))
if opts['save_results']:
    print("Saving results...")
    save_dict(join(output_folder, opts['exp_name'] + "_metrics_train.json"), metrics_train)
    save_dict(join(output_folder, opts['exp_name'] + "_metrics_val.json"), metrics_val)
    save_dict(join(output_folder, opts['exp_name'] + "_metrics_test.json"), metrics_test)
print("[Elapsed: " + elapsed_time(start_time, time.time()) + "]")
