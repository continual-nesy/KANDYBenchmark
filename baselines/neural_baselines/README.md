# Neural baselines

## Usage
```commandline
python main.py parameters
```

Parameter list:

| Argument            | Values                                                          | Default  | Description                                                                  |
|---------------------|-----------------------------------------------------------------|----------|------------------------------------------------------------------------------|
| `--data_path`       | str                                                             | `./`     | Root of the data folder                                                      |
| `--model`           | `mlp`, `cnn`, `resnet50`, `resnet50_head_only`, `vit_head_only` | `mlp`    | Neural network to train                                                      |
| `--train`           | `independent`, `joint`, `continual_task`, `continual_online`    | `joint`  | Training scheme                                                              |
| `--supervised_only` | bool                                                            | `False`  | Consider only supervised samples                                             |
| `--augment`         | bool                                                            | `True`   | Perform random augmentation of the training data                             |
| `--lr`              | float                                                           | `-0.001` | Learning rate. If < 0 use Adam, otherwise use SGD                            |
| `--weight_deca`     | float                                                           | `0.0`    | Weight decay factor                                                          |
| `--batch`           | int                                                             | `16`     | Minibatch size                                                               |
| `--task_epochs`     | int                                                             | `1`      | Number of epochs for each task. Incompatible with `--train continual_online` |
| `--balance`         | bool                                                            | `False`  | Resample positives and negatives to achieve balanced training data           |
| `--replay_buffer`   | int                                                             | `0`      | Size of the replay buffer. Only with `--train continual_*`                   |
| `--replay_lambda`   | float                                                           | `0.0`    | Weight of experience replay loss. Only with `--train_continual_*`            |
| `--seed`            | int                                                             | `-1`     | Seed for random generator. If < 0 use system time                            |
| `--output_folder`   | str                                                             | `exp`    | Output folder                                                                |
| `--device`          | str                                                             | `cpu`    | Torch device for experiments                                                 |
| `--save_net`        | bool                                                            | `True`   | Save network weights at the end of the experiment                            |
| `--save_results`    | bool                                                            | `True`   | Save results at the end of the experiment                                    |
| `--save_options`    | bool                                                            | `True`   | Save options at the beginning of the experiment                              |
| `--print_every`     | int                                                             | `10`     | Number of gradient steps before consecutive prints                           |
| `--wandb_project`   | str                                                             | `None`   | W&B project name to optionally log results to                                |
| `--wandb_group`     | str                                                             | `None`   | Group within the W&B project                                                 |
