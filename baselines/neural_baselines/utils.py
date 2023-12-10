import json
import torch
import time
import numpy as np
import random
from collections import OrderedDict
from argparse import ArgumentTypeError
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import TaskOrganizedDataset


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
            raise ArgumentTypeError("Invalid number type (it must be int or float)")
        if not ((self.__min is None and self.__max is None) or
                (self.__max is None) or (self.__min is not None and self.__min < self.__max)):
            raise ArgumentTypeError("Invalid range")

    def __call__(self, value: int | float | str) -> int | float:
        try:
            val = self.__number_type(value)
        except ValueError:
            raise ArgumentTypeError(f"Invalid value specified, conversion issues! Provided: {value}")
        if self.__min is not None and val < self.__min:
            raise ArgumentTypeError(f"Invalid value specified, it must be >= {self.__min}")
        if self.__max is not None and val > self.__max:
            raise ArgumentTypeError(f"Invalid value specified, it must be <= {self.__max}")
        return val


class ArgBoolean:
    """Implement the notion of 'number' when passed as input argument to the program, deeply checking it.
    It allows the user to provide it in different forms {1, 'True', 'true', 'yes'}, {0, 'False', 'false', 'no'}."""

    def __call__(self, value: str | bool | int) -> bool:
        if isinstance(value, str):
            val = value.lower().strip()
            if val != "true" and val != "false" and val != "yes" and val != "no":
                raise ArgumentTypeError(f"Invalid value specified: {value}")
            val = True if (val == "true" or val == "yes") else False
        elif isinstance(value, int):
            if value != 0 and value != 1:
                raise ArgumentTypeError(f"Invalid value specified: {value}")
            val = value == 1
        elif isinstance(value, bool):
            val = value
        else:
            raise ArgumentTypeError(f"Invalid value specified (expected boolean): {value}")
        return val


def generate_experiment_name(prefix: str | None = None, suffix: str | None = None) -> str:
    """Generate a dummy name for an experiment, such as 2023-08-10_17-06-22_Columbus.

        :param prefix: An optional prefix to add to the experiment name.
        :param suffix: An optional suffix to add to the experiment name (by default, it is a city from Ohio, USA).
        :returns: A string that represents the name of an experiment, such as 2023-08-10_17-06-22_Columbus.
    """

    def_suffix = ["Akron", "Alliance", "Ashtabula", "Athens", "Barberton", "Bedford", "Bellefontaine", "Canton",
                  "Chillicothe", "Cincinnati", "Cleveland", "Columbus", "Conneaut", "Dayton", "Defiance", "Delaware",
                  "Elyria", "Euclid", "Findlay", "Gallipolis", "Greenville", "Hamilton", "Kent", "Kettering",
                  "Lakewood", "Lancaster", "Lima", "Lorain", "Mansfield", "Marietta", "Marion", "Massillon", "Mentor",
                  "Middletown", "Milan", "Newark", "Niles", "Norwalk", "Oberlin", "Painesville", "Parma", "Piqua",
                  "Portsmouth", "Salem", "Sandusky", "Springfield", "Steubenville", "Tiffin", "Toledo", "Urbana",
                  "Warren", "Wooster", "Worthington", "Xenia", "Youngstown", "Zanesville"]
    ret = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if prefix is not None:
        ret = prefix + "_" + ret
    if suffix is not None:
        ret = ret + "_" + suffix
    else:
        ret = ret + "_" + random.choice(def_suffix)
    return ret


def load_dict(json_file_path: str) -> dict:
    """Load a JSON dict from file.

        :param json_file_path: The path to the file.
        :returns: A Python dictionary with the JSON contents.
    """

    f = open(json_file_path, "r")
    if f is None or not f or f.closed:
        raise IOError("Error while loading data. Cannot read: " + json_file_path)
    json_loaded = json.load(f)
    f.close()
    return json_loaded


def save_dict(json_file_path: str,
              dict_to_save: dict,
              keys_to_exclude : list | tuple = None,
              one_lined: bool = False):
    """Save a Python dictionary to a JSON file.

        :param json_file_path: The path to the destination file.
        :param dict_to_save: The dictionary to be saved.
        :param keys_to_exclude: A list/tuple of keys to discard while saving.
        :param one_lined: If True, the whole dictionary will be formatted on a single line of the destination file.
    """

    f = open(json_file_path, "w")
    if f is None or not f or f.closed:
        raise IOError("Error while saving data. Cannot access: " + json_file_path)
    filtered_dict = {}
    filtered_structured_dict = {}
    for k, v in dict_to_save.items():
        if keys_to_exclude is None or not (k in keys_to_exclude):
            if not (v is None or isinstance(v, int) or isinstance(v, float) or isinstance(v, str)):
                filtered_structured_dict[k] = v
            else:
                filtered_dict[k] = v
    d1 = OrderedDict(sorted(filtered_dict.items()))
    d2 = OrderedDict(sorted(filtered_structured_dict.items()))
    d1.update(d2)
    if not one_lined:
        json.dump(d1, f, indent=4)
    else:
        f.write(json.dumps(d1).replace("], ", "],\n").replace("{", "{\n").replace("}", "\n}\n"))
    f.close()


def set_seed(seed: int):
    """Set the seed of random number generators in a paranoid manner (Pytorch).

        :param seed: An integer seed (if -1, then the current time is used).
    """

    seed = int(time.time()) if seed < 0 else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def accuracy(o: torch.Tensor, y: torch.Tensor) -> float:
    """Classic (per-class-balanced) accuracy in [0,1].

        :param o: Tensor with output predictions.
        :param y: Tensor with expected targets.
        :returns: Accuracy score in [0,1] (average of per-class accuracies).
    """

    assert (o.ndim == 2 and y.ndim == 2) or (o.ndim == 1 and y.ndim == 1), "Unsupported tensor shapes."

    acc = 0.
    y_unique, idx = torch.unique(y, dim=0, return_inverse=True)
    num_classes = y_unique.numel()
    for i in range(0, num_classes):
        _idx = idx == i
        if o.ndim == 2:
            acc += torch.mean((torch.eq(o[_idx, :], y[_idx, :])).to(torch.float)).item()
        else:
            acc += torch.mean((torch.eq(o[_idx], y[_idx])).to(torch.float)).item()
    return acc / num_classes


def avg_accuracy(acc_matrix: torch.Tensor) -> float:
    """Compute the average accuracy on a multi-task dataset, with sequentially provided tasks.

        :param acc_matrix: The num_task-by-num_task matrix, where the (i,j)-th entry is the accuracy of the model
            trained up task i, evaluated on data from task j.
        :returns: The average accuracy of the most recent model (mean of the last line of such a matrix).
    """

    return torch.mean(acc_matrix[-1, :]).item()


def avg_forgetting(acc_matrix: torch.Tensor) -> float:
    """Compute the average forgetting on a multi-task dataset, with sequentially provided tasks.

        :param acc_matrix: The num_task-by-num_task matrix, where the (i,j)-th entry is the accuracy of the model
            trained up task i, evaluated on data from task j.
        :returns: The average forgetting score.
    """

    if acc_matrix.shape[0] == 1:
        return 0.
    else:
        _acc_star, _ = torch.max(acc_matrix[0:-1, 0:-1], dim=0)  # discarding last row and last column
        _acc = acc_matrix[-1, 0:-1]  # this is the last row, discarding the last column
        _forg = torch.mean(_acc_star - _acc).item()
        return _forg


def backward_transfer(acc_matrix: torch.Tensor) -> float:
    """Compute the positive backward transfer on a multi-task dataset, with sequentially provided tasks.

        :param acc_matrix: The num_task-by-num_task matrix, where the (i,j)-th entry is the accuracy of the model
            trained up task i, evaluated on data from task j.
        :returns: The positive backward transfer score.
    """

    if acc_matrix.shape[0] == 1:
        return 0.
    else:
        _CCM_diff = acc_matrix - torch.diag(acc_matrix)
        _bwt = (2. * torch.sum(torch.tril(_CCM_diff, diagonal=-1))) / float(acc_matrix.shape[0] *
                                                                            (acc_matrix.shape[0] - 1))
        return max(_bwt.item(), 0.)


def forward_transfer(acc_matrix: torch.Tensor) -> float:
    """Compute the forward transfer on a multi-task dataset, with sequentially provided tasks.

        :param acc_matrix: The num_task-by-num_task matrix, where the (i,j)-th entry is the accuracy of the model
            trained up task i, evaluated on data from task j.
        :returns: The forward transfer score.
    """

    if acc_matrix.shape[0] == 1:
        return 0.
    else:
        _fwd = (2. * torch.sum(torch.triu(acc_matrix, diagonal=1))) / float(acc_matrix.shape[0] *
                                                                            (acc_matrix.shape[0] - 1))
        return _fwd.item()


def compute_accuracies(net: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module],
                       dataset: TaskOrganizedDataset,
                       batch_size: int = 16,
                       device: str = 'cpu',
                       tune_decision_thresholds: bool = False,
                       tune_last_task_only: bool = False) -> list[float]:
    """Compute the accuracy on a given dataset, returning the task-specific accuracies in [0,1].

        :param net: Net or list/tuple of neural networks (a net for all tasks, or a net for each tasks).
        :param dataset: The dataset to consider in the evaluation (it can be composed by multiple tasks).
        :param batch_size: The size of mini-batch for evaluation purposes.
        :param device: The string with the device name on which computations will be performed ('cpu', 'cuda:0', ...).
        :param tune_decision_thresholds: If True, then the network decision threshold (initially 0.5) will be tuned.
        :param tune_last_task_only: If 'tune_decision_thresholds' is True and this param is true as well,
            only the decision threshold of the predictor associated to the last task will be tuned.
        :returns: The list of task-specific accuracies, each of them in [0,1] (list of T elements for T tasks).
    """

    with torch.no_grad():

        # splitting dataset in function of the task ID
        task_datasets = dataset.get_task_datasets()

        # basic containers
        accuracies_per_task = []
        device = torch.device(device)
        one_net_per_task = isinstance(net, (tuple, list))
        independent_nets = net if one_net_per_task else None

        assert not one_net_per_task or len(task_datasets) == len(independent_nets), \
            "The number of tasks must be equal to the number of nets."

        # looping on the datasets of each task
        for task_id, task_dataset in enumerate(task_datasets):

            # selecting the right network
            net = independent_nets[task_id] if one_net_per_task else net
            net.to(device)
            net.eval()

            # selecting the right decision threshold
            decision_threshold = net.decision_thresholds[task_id] \
                if net.decision_thresholds.numel() > 1 else net.decision_thresholds[0]
            decision_threshold = decision_threshold.cpu()

            # preparing data access
            data_loader = DataLoader(task_dataset, batch_size=batch_size, shuffle=False)
            o = []
            y = []

            # looping on data
            for (_x, _y, _, _, _) in data_loader:

                # predicting and saving result (cpu)
                _o = torch.sigmoid(net(_x.to(device)))  # sigmoid here, warning!
                o.append(_o[:, task_id].cpu() if _o.shape[1] > 1 else _o[:, 0].cpu())
                y.append(_y)

            # merging results
            o = torch.concat(o, dim=0)
            y = torch.concat(y, dim=0)

            # computing accuracies, possibly tuning the decision threshold to maximize the accuracy on each task
            if tune_decision_thresholds and (not tune_last_task_only or task_id == len(task_datasets) - 1):
                thresholds = [0.5, 0.4, 0.6, 0.3, 0.7]
                best_acc = -1.
                best_threshold = None

                for t in thresholds:
                    acc = accuracy((o > t).to(y.dtype), y)
                    if acc > best_acc:
                        best_threshold = t
                        best_acc = acc

                if net.decision_thresholds.numel() > 1:
                    net.decision_thresholds[task_id] = best_threshold
                else:
                    net.decision_thresholds[0] = best_threshold
            else:
                best_acc = accuracy((o > decision_threshold).to(y.dtype), y)

            # storing accuracies
            accuracies_per_task.append(best_acc)  # per-task

            # moving the network back to CPU to free GPU memory, if needed
            if one_net_per_task:
                net.cpu()
        return accuracies_per_task


# print metrics (named with string _name) computed right after having processed a given distribution (_distribution)
def print_metrics(metrics: dict, tasks_seen_so_far: int) -> None:
    """Print a pre-allocated and partially filled metrics dictionary, limiting it to the tasks considered so far.

        :param metrics: A metrics dictionary (fields are: 'acc_matrix', 'avg_accuracy', 'avg_forgetting',
            'backward_transfer', 'forward_transfer'). Each field stores either a pre-allocated matrix ('acc_matrix'),
            or a pre-allocated vector with per-task-scores (all the other fields).
        :param tasks_seen_so_far: The number of tasks considered so far, so that printing will only consider them.
    """

    def format_acc_matrix(acc_matrix):
        _s = "       "
        for j in range(0, acc_matrix.shape[1]):
            _s += " | {:3d}".format(j) + " "
        _s += '\n'
        _s += "    ----"
        for j in range(0, acc_matrix.shape[1]):
            _s += "+------"
        _s += '\n'
        for i in range(0, acc_matrix.shape[0]):
            _s += "   {:3d}".format(i) + " "
            for j in range(0, acc_matrix.shape[1]):
                a = acc_matrix[i][j].item()
                _s += " | {:.2f}".format(a) if a != -1 else " | N.A."
            _s += '\n' if i < acc_matrix.shape[0] - 1 else ''
        return _s

    s = "[" + metrics['name'] + "] Metrics after " + str(tasks_seen_so_far) + " training tasks"
    s += "\n    acc_matrix:\n"
    s += format_acc_matrix(metrics['acc_matrix']) + "\n"
    s += "    avg_accuracy: {:.2f}".format(metrics['avg_accuracy'][tasks_seen_so_far - 1]) + ", "
    s += "avg_forgetting: {:.2f}".format(metrics['avg_forgetting'][tasks_seen_so_far - 1]) + ", "
    s += "backward_transfer: {:.2f}".format(metrics['backward_transfer'][tasks_seen_so_far - 1]) + ", "
    s += "forward_transfer: {:.2f}".format(metrics['forward_transfer'][tasks_seen_so_far - 1])
    print(s)
