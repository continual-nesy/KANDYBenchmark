# ILP baselines

## Usage

```commandline
python main.py params

```

Parameter list:

| Argument                | Values               | Default                 | Description                                                                                                                         |
|-------------------------|----------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `--output_folder`       | str                  | `exp`                   | Output folder                                                                                                                       |
| `--save_options`        | bool                 | `True`                  | Save results at the end of the experiment                                                                                           |
| `--wandb_project`       | str                  | `None`                  | W&B project name to optionally log results to                                                                                       |
| `--wandb_group`         | str                  | `None`                  | Group within the W&B project                                                                                                        |
| `--csv_dir`             | str                  | `.`                     | Annotaton folder                                                                                                                    |
| `--train_csv`           | str                  | `train_annotations.csv` | Training annotations                                                                                                                |
| `--val_csv`             | str                  | `None`                  | Validation annotations                                                                                                              |
| `--test_csv`            | str                  | `None`                  | Test annotations                                                                                                                    |
| `--prefix`              | str                  | `minimal`               | Background knowledge and bias prefix (looking for `_{natural,pointer}_bg.pl` and `_{natural,pointer}_{popper,aleph}_bias.pl` files) |
| `--prefix_cheat`        | str                  | `None`                  | Cheat predicates prefix (looking for _{natural,pointer}_bg.pl and _{natural,pointer}_{popper,aleph}_bias.pl files)                  |
| `--engine`              | `aleph`, `popper`    | `popper`                | ILP engine to use                                                                                                                   |
| `--encoding`            | `natural`, `pointer` | `natural`               | Encoding to use                                                                                                                     |
| `--timeout`             | int                  | `600`                   | Timeout (in seconds) for a single task                                                                                              |
| `--max_vars`            | int                  | `6`                     | Maximum number of variables in the body of a clause                                                                                 |
| `--max_clauses`         | int                  | `1`                     | Maximum number of clauses in a theory                                                                                               |
| `--max_size`            | int                  | `6`                     | Maximum number of literals in the body of a clause                                                                                  |
| `--singleton_vars`      | bool                 | `False`                 | Enable singleton variables. Only with `--engine popper`                                                                             |
| `--predicate_invention` | bool                 | `False`                 | Enable predicate invention. Only with `--engine popper`                                                                             |
| `--recursion`           | bool                 | `False`                 | Enable recursive clauses. Only with `--engine popper`                                                                               |
| `--max_literals`        | int                  | `40`                    | Maximum number of literals in the entire theory. Only with `--engine popper`                                                        |
| `--min_acc`             | float                | `0.5`                   | Minimum acceptable accuracy. Only with `--engine aleph`                                                                             |
| `--noise`               | int                  | `0`                     | Maximum acceptable number of false positives. Only with `--engine aleph`                                                            |
| `--i`                   | int                  | `2`                     | Layers of new variables. Only with `--engine aleph`                                                                                 |
| `--min_pos`             | int                  | `2`                     | Minimum number of positives covered by a clause. Only with `--engine aleph`                                                         |
| `--depth`               | int                  | `10`                    | Proof depth. Only with `--engine aleph`                                                                                             |
| `--nodes`               | int                  | `5000`                  | Nodes to explore during search. Only with `--engine aleph`                                                                          |

## Aleph license
```prolog
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A Learning Engine for Proposing Hypotheses                              %
%                                                                         %
% A L E P H                                                               %
% Version 5    (last modified: Sun Mar 11 03:25:37 UTC 2007)              %
%                                                                         %
% This is the source for Aleph written and maintained                     %
% by Ashwin Srinivasan (ashwin@comlab.ox.ac.uk)                           %
%                                                                         %
%                                                                         %
% It was originally written to run with the Yap Prolog Compiler           %
% Yap can be found at: http://sourceforge.net/projects/yap/               %
% Yap must be compiled with -DDEPTH_LIMIT=1                               %
%                                                                         %
% It should also run with SWI Prolog, although performance may be         %
% sub-optimal.                                                            %
%                                                                         %
% If you obtain this version of Aleph and have not already done so        %
% please subscribe to the Aleph mailing list. You can do this by          %
% mailing majordomo@comlab.ox.ac.uk with the following command in the     %
% body of the mail message: subscribe aleph                               %
%                                                                         %
% Aleph is freely available for academic purposes.                        %
% If you intend to use it for commercial purposes then                    %
% please contact Ashwin Srinivasan first.                                 %
%                                                                         %
% A simple on-line manual is available on the Web at                      %
% www.comlab.ox.ac.uk/oucl/research/areas/machlearn/Aleph/index.html      %
%                                                                         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```
## Popper license

MIT License

Copyright (c) 2021 Logic and Learning lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.