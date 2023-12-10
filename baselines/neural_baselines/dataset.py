import os
import random
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import os.path as path
import glob
from typing import Any


class TaskOrganizedDataset(Dataset):
    """This class implements a Pytorch Dataset that abstracts a task-organized data-folder structure."""

    def __init__(self,
                 data_folder: str,
                 task_ids: tuple[int, ...] | list[int, ...] | None = None,
                 supervised_only: bool = False,
                 transform: Any | None = None,
                 target_transform: Any | None = None,
                 max_buffer_size: int = 0) -> None:
        """Initialize the class.

            :param data_folder: The path to the root of the data (i.e., where the task-sub-folders are located).
            :param task_ids: A tuple/list of task IDs to consider
                (optional - if None, all tasks are considered).
            :param supervised_only: Boolean flag to consider supervised data only (default: False).
            :param transform: The transformations to apply to each images (optional).
            :param target_transform: The transformations to apply to each target (optional).
            :param max_buffer_size: Size of the examples to buffer (default: 0).
        """
        self.data_folder = data_folder
        self.task_ids = task_ids  # it can be None
        self.supervised_only = supervised_only
        self.transform = transform
        self.target_transform = target_transform

        self.buffered_indices = []
        self.samples_seen_so_far = 0
        self.task2buffered_positives = {}
        self.task2buffered_negatives = {}
        self.max_buffer_size = max_buffer_size

        # loading annotations from "annotations.csv" (considering only the required tasks, self.task_ids)
        self.annotations, self.task2num_examples, self.task2num_supervised, \
            self.task2positive_examples, self.task2negative_examples = self.__load_annotations()

        # updating task IDs (it could have been None at the beginning)
        self.task_ids = list(self.task2num_examples.keys())
        self.task_ids.sort()
        self.num_tasks = len(self.task_ids)

        # mapping task to incremental, continuous, integers
        self.task2zero_based_index = {}
        for i, task_id in enumerate(self.task_ids):
            self.task2zero_based_index[task_id] = i

        # guessing shape of the input data
        image, _, _, _, _ = self[0]
        self.input_shape = image.shape

        # cache
        self.task_datasets = None  # list of per-task dataset (it gets populated only if get_task_datasets() is called)

    def __len__(self) -> int:
        """Return the number of examples collected in this dataset.

            :returns: The number of examples in this dataset.
        """

        return len(self.annotations.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int, int]:
        """Get the idx-th element of the dataset.

            :param idx: Integer index of the element to get.
            :returns: A torch.Tensor (torch.uint8) without further rescaling; an integer label; the task ID;
                the sample ID.
        """
        annotation = self.annotations.iloc[idx]
        image = read_image(os.path.join(self.data_folder, annotation['filename']))
        label = annotation['label'] if annotation['supervised'] else None
        task_id = annotation['task_id']
        zero_based_task_id = self.task2zero_based_index[task_id]
        absolute_id = annotation['id']

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, task_id, zero_based_task_id, absolute_id

    def __str__(self) -> str:
        """Collect dataset stats into a string.

            :returns: The string with all the information about this dataset.
        """

        s = ""
        s += "[TaskOrientedDataset]"
        s += "\n- data folder: " + self.data_folder
        s += "\n- total number of examples: " + str(len(self))
        s += "\n- number of tasks: " + str(self.num_tasks)
        s += "\n- task IDs: " + str(self.task_ids)
        s += "\n- task ID -> number of examples:"
        for task_id in self.task_ids:
            s += "\n    " + str(task_id) + " -> " + str(self.task2num_examples[task_id]) + \
                 (" (out of which " + str(self.task2num_supervised[task_id]) + " are supervised)"
                  if self.task2num_supervised[task_id] < self.task2num_examples[task_id] else
                  " (all of them are supervised)")
        s += "\n- task ID -> number of positive examples:"
        for task_id in self.task_ids:
            s += "\n    " + str(task_id) + " -> " + str(len(self.task2positive_examples[task_id]))
        return s

    def get_buffered_sample_indices(self) -> list[int]:
        """Get the list of indices of buffered samples.

            :returns: A list of indices.
        """

        return self.buffered_indices

    def buffer_sample(self, idx: int, balanced: bool = False):
        """Possibly add a new example index to the list of buffered samples (reservoir sampling).

            :param idx: The index of the example.
            :param balanced: Boolean flag to indicate if the buffer should be balanced (tasks and positives/negatives).
            :returns: True if the sample was added to the buffer (False, otherwise).
        """

        # retrieving the label and task ID of the the provided example
        annotation = self.annotations.iloc[idx]
        label = annotation['label'] if annotation['supervised'] else None
        task_id = annotation['task_id']
        added = False

        # reservoir-sampling-based storage
        if self.max_buffer_size > self.samples_seen_so_far:

            # saving the given example to the buffer
            j = len(self.buffered_indices)
            self.buffered_indices.append(idx)
            added = True
        else:
            j = random.randint(0, self.samples_seen_so_far)
            if j < len(self.buffered_indices):

                # determining the example of the buffer to be removed
                if balanced:
                    max_positives = 0
                    task_with_largest_positives = -1

                    for _task_id, _buffered_indices in self.task2buffered_positives.items():
                        if len(_buffered_indices) > max_positives:
                            max_positives = len(_buffered_indices)
                            task_with_largest_positives = _task_id

                    max_negatives = 0
                    task_with_largest_negatives = -1

                    for _task_id, _buffered_indices in self.task2buffered_negatives.items():
                        if len(_buffered_indices) > max_negatives:
                            max_negatives = len(_buffered_indices)
                            task_with_largest_negatives = _task_id

                    if max_positives > max_negatives:
                        j = random.choice(self.task2buffered_positives[task_with_largest_positives])
                    else:
                        j = random.choice(self.task2buffered_negatives[task_with_largest_negatives])

                # removing an example from the buffer
                old_annotation = self.annotations.iloc[self.buffered_indices[j]]
                old_label = old_annotation['label'] if old_annotation['supervised'] else None
                old_task_id = old_annotation['task_id']
                if old_label == 1:
                    self.task2buffered_positives[old_task_id].remove(j)
                    if self.task2buffered_positives[old_task_id] is None or \
                            len(self.task2buffered_positives[old_task_id]) == 0:
                        del self.task2buffered_positives[old_task_id]
                else:
                    self.task2buffered_negatives[old_task_id].remove(j)
                    if self.task2buffered_negatives[old_task_id] is None or \
                            len(self.task2buffered_negatives[old_task_id]) == 0:
                        del self.task2buffered_negatives[old_task_id]

                # saving the given example to the buffer
                self.buffered_indices[j] = idx
                added = True

        # recording the number of examples that were considered for buffering purposes so far
        self.samples_seen_so_far += 1

        # updating stats
        if added:
            if label == 1:
                if task_id not in self.task2buffered_positives:
                    self.task2buffered_positives[task_id] = [j]
                else:
                    self.task2buffered_positives[task_id].append(j)
            else:
                if task_id not in self.task2buffered_negatives:
                    self.task2buffered_negatives[task_id] = [j]
                else:
                    self.task2buffered_negatives[task_id].append(j)

        return added

    def get_balanced_sample_indices(self) -> list[int]:
        """Get a list of indices of data samples that ara balanced with respect to positive and negative classes.

            :returns: A list of indices.
        """

        indices = []

        for task_id in self.task_ids:
            task2neg = self.task2negative_examples[task_id]
            task2pos = self.task2positive_examples[task_id]

            if len(task2neg) == 0:
                indices.extend(task2pos)
                continue
            if len(task2pos) == 0:
                indices.extend(task2neg)
                continue

            if len(task2neg) == len(task2pos):
                indices.extend(task2pos)
                indices.extend(task2neg)
            elif len(task2pos) > len(task2neg):
                indices.extend(task2pos)
                indices.extend(task2neg)
                diff = len(task2pos) - len(task2neg)
                n = len(task2neg)
                for i in range(0, diff):
                    indices.append(task2neg[i % n])
            else:
                indices.extend(task2pos)
                indices.extend(task2neg)
                diff = len(task2neg) - len(task2pos)
                n = len(task2pos)
                for i in range(0, diff):
                    indices.append(task2pos[i % n])

        return indices

    def get_task_datasets(self) -> list[Dataset]:
        """Return a list of TaskOrientedDataset, one for each task of the current dataset.

            :returns: A list of TaskOrientedDataset, one for each existing task of the current dataset.
        """

        if self.task_datasets is None:
            self.task_datasets = []
            for task_id in self.task_ids:
                single_task_dataset = TaskOrganizedDataset(self.data_folder,
                                                           task_ids=(task_id,),
                                                           supervised_only=self.supervised_only,
                                                           transform=self.transform,
                                                           target_transform=self.target_transform)
                self.task_datasets.append(single_task_dataset)
        return self.task_datasets

    def __load_annotations(self) -> \
            tuple[pd.DataFrame, dict[int, int], dict[int, int], dict[int, list[int]], dict[int, list[int]]]:
        """Load and check the contents of the annotations.csv file in the data folder, ensuring data is present.

            :returns: Annotations loaded into a (cleaned) Pandas DataFrame; a dictionary mapping each task ID to
                the number of examples for such a task; a dictionary mapping each task ID to the number of supervised
                examples for such a task; a dictionary mapping each task ID to the idx of positive examples;
                a dictionary mapping each task ID to the idx of a negative examples.
        """

        csv_file = path.join(self.data_folder, "annotations.csv")
        assert path.exists(csv_file), "Invalid data folder (cannot find annotations.csv): " + self.data_folder
        ann = pd.read_csv(csv_file, dtype={'filename': 'str',
                                           'task_id': 'int',
                                           'label': 'int',
                                           'supervised': 'bool'})

        # checking format
        assert 'filename' in ann.columns, "Cannot find 'filename' column in " + csv_file
        assert 'task_id' in ann.columns, "Cannot find 'task_id' column in " + csv_file
        assert 'label' in ann.columns, "Cannot find 'label' column in " + csv_file
        assert 'supervised' in ann.columns, "Cannot find 'supervised' column in " + csv_file

        # removing unused columns
        ann = ann[['filename', 'task_id', 'label', 'supervised']]
        filename_col_idx = ann.columns.get_loc('filename')

        # adding unique sample identifier
        ann['id'] = list(range(0, len(ann)))

        # ensuring each row is fine
        tasks_in_csv = {}
        for i in range(len(ann)):
            row = ann.iloc[i]
            filename = row["filename"]
            task_id = row["task_id"]
            label = row["label"]
            supervised = row["supervised"]

            assert task_id is not None and task_id >= 0, "Invalid (negative) task ID " \
                                                         "at row " + str(i + 1) + " of " + csv_file
            assert label is not None and label in [0, 1], "Invalid label at row " + str(i + 1) + " of " + csv_file
            assert supervised is not None, "Invalid supervised-flag at row " + str(i + 1) + " of " + csv_file
            assert filename is not None and filename[0:len(str(task_id))] == str(task_id) \
                and (filename[len(str(task_id))] == '/' or filename[len(str(task_id))] == '\\') \
                and filename[len(str(task_id)) + 1].isdigit(), \
                "Invalid filename format (expected 'task_id/file' or 'task_id\\file, with 'file' starting " \
                "with a digit) at row " + str(i + 1) + " of " + csv_file

            tasks_in_csv[task_id] = True

        # checking coherence between the information from annotations.csv and the file/folder structure/contents
        assert len(next(os.walk(self.data_folder))[1]) == len(tasks_in_csv), \
            "The number of task folders does not match the number of tasks in " + csv_file

        # removing unneeded rows (if needed)
        if self.task_ids is not None:
            for task_id in self.task_ids:
                assert task_id in tasks_in_csv, \
                    "The provided task ID (" + str(task_id) + ") does not exist (task IDs are: " + \
                    str(tasks_in_csv) + ")."

            ann = ann[ann['task_id'].isin(self.task_ids)]

        # cleaning the kept rows, counting, checking file system
        task2num_examples = {}
        task2num_supervised = {}
        task2positive_examples = {}
        task2negative_examples = {}
        for i in range(len(ann)):
            row = ann.iloc[i]
            filename = row["filename"]
            task_id = row["task_id"]
            supervised = row["supervised"]
            label = row["label"]

            # re-encoding the filename with the right-os-dependent separator
            filename = filename[len(str(task_id)) + 1:]
            filename = path.join(str(task_id), filename)
            ann.iloc[i, filename_col_idx] = filename

            # counting examples per task
            task2num_examples[task_id] = 1 if task_id not in task2num_examples else task2num_examples[task_id] + 1
            if task_id not in task2positive_examples:
                task2positive_examples[task_id] = []
                task2negative_examples[task_id] = []
            if label == 1:
                task2positive_examples[task_id].append(i)
            else:
                task2negative_examples[task_id].append(i)
            task2num_supervised[task_id] = int(supervised) if task_id not in task2num_supervised else \
                task2num_supervised[task_id] + int(supervised)

            # ensuring the annotated file exists
            assert path.isfile(path.join(self.data_folder, filename)), \
                "Not-existing filename at row " + str(i + 1) + " of " + csv_file

        for task_id in task2num_examples.keys():
            assert path.isdir(path.join(self.data_folder, str(task_id))), \
                "Cannot find folder to task " + str(task_id) + " (expected: " + \
                path.join(self.data_folder, str(task_id)) + ")"

            png_files = glob.glob(path.join(self.data_folder, str(task_id), "*.png"))

            assert len(png_files) == task2num_examples[task_id], \
                "The number of task examples for task " + str(task_id) + " in " + csv_file + \
                " does not match the number of files in " + \
                self.data_folder + " (they are " + str(task2num_examples[task_id]) + " and " + \
                str(len(png_files)) + ", respectively)"

        return ann, task2num_examples, task2num_supervised, task2positive_examples, task2negative_examples


def check_data_folder(data_path: str) -> None:
    """Check the structure of the data folder (the one where the 'train' sub-folder is located).

        :param data_path: Path to the data folder root.
    """

    # checking folder structure
    assert path.isdir(data_path), "Not-existing data path " + str(data_path) + " (expected: existing folder)."

    # main sub-folders
    assert path.isdir(path.join(data_path, "train")), "Missing sub-folder with training data ('train')."
    assert path.isdir(path.join(data_path, "val")), "Missing sub-folder with validation data ('val')."
    assert path.isdir(path.join(data_path, "test")), "Missing sub-folder with test data ('test')."
    assert path.isfile(path.join(data_path, "stats.txt")), "Missing specs file ('specs.txt')."

    # determining folder format (two types of folder organizations)
    folder_format = ['curriculum_like', 'task_sub_folders']
    if path.isfile(path.join(data_path, 'train_annotations.csv')):
        folder_format = folder_format[0]
        assert path.isfile(path.join(data_path, 'val_annotations.csv')), "Cannot find val_annotations.csv"
        assert path.isfile(path.join(data_path, 'test_annotations.csv')), "Cannot find test_annotations.csv"
    else:
        folder_format = folder_format[1]

    # counting and checking task-related folders (task-sub-folders)
    if folder_format == 'task_sub_folders':
        tasks = [[], [], []]
        for folder_id, p in enumerate([path.join(data_path, "train"),
                                       path.join(data_path, "val"),
                                       path.join(data_path, "test")]):
            contents = os.listdir(p)
            found_annotations = False

            # ensure annotations.csv is there
            for c in contents:
                if not found_annotations and c == "annotations.csv" and path.isfile(path.join(p, c)):
                    found_annotations = True
                if path.isdir(path.join(p, c)):
                    assert c.isdecimal() and (c[0] != '0' or len(c) == 1), \
                        "Invalid folder " + path.join(p, c) + \
                        " (expected task folder, named with a non-negative integer)-"
                    tasks[folder_id].append(c)

            assert found_annotations, "Cannot find annotations.csv in folder " + p

        tasks[0].sort()
        tasks[1].sort()
        tasks[2].sort()

        assert tasks[0] == tasks[1] and tasks[1] == tasks[2], \
            "Mismatching task sub-folders among train, val, test sets."

    # counting and checking task-related folders (curriculum-like folder)
    elif folder_format == 'curriculum_like':
        raise ValueError("Unsupported data folder format (curriculum-like). "
                         "Expected train, test, val folders with annotations.csv in each of them.")
    else:
        raise ValueError("Something went wrong, this was not supposed to happen at all :)")
