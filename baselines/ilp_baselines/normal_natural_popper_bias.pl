head_pred(valid, 1).

%body_pred(red, 1).
%body_pred(green, 1).
%body_pred(blue, 1).
%body_pred(cyan, 1).
%body_pred(magenta, 1).
%body_pred(yellow, 1).
%body_pred(small, 1).
%body_pred(large, 1).
%body_pred(triangle, 1).
%body_pred(circle, 1).
%body_pred(square, 1).

%body_pred(random, 1).
%body_pred(stack, 1).
%body_pred(grid, 1).
%body_pred(stack_reduce_bb, 1).
%body_pred(side_by_side, 1).
%body_pred(side_by_side_reduce_bb, 1).
%body_pred(diag_ul_lr, 1).
%body_pred(diag_ll_ur, 1).

%body_pred(quadrant_ul, 1).
%body_pred(quadrant_ur, 1).
%body_pred(quadrant_ll, 1).
%body_pred(quadrant_lr, 1).


%body_pred(color, 1).
%body_pred(shape, 1).
%body_pred(size, 1).

%body_pred(non_diag, 1).
%body_pred(diag, 1).

%body_pred(non_random, 1).
%body_pred(any_composition, 1).
%body_pred(line, 1).

%body_pred(quadrant, 1).
%body_pred(quadrant_or_center, 1).

body_pred(extract_shape, 2).
body_pred(extract_color, 2).
body_pred(extract_size, 2).

body_pred(exists_shape, 2).
body_pred(exists_color, 2).
body_pred(exists_size, 2).

body_pred(same_shape, 2).
body_pred(same_color, 2).
body_pred(same_size, 2).

body_pred(contains, 2).
body_pred(recursive_contains, 2).
body_pred(contains_composition, 2).
body_pred(contains_composition_depth, 3).

body_pred(extract_operator, 2).
body_pred(extract_children, 2).
body_pred(extract_op_and_chld, 3).

body_pred(same_attribute, 1).
body_pred(same_non_size, 1).
body_pred(all_same, 1).

body_pred(first, 2).
body_pred(last, 2).
body_pred(prepend, 3).
body_pred(droplast, 2).
body_pred(middle, 2).
body_pred(getmiddle, 2).
body_pred(dropmiddle, 2).

body_pred(less, 2).
body_pred(less_eq, 2).
body_pred(greater, 2).
body_pred(same_int, 2).
body_pred(different_int, 2).
body_pred(same_obj, 2).
body_pred(different_obj, 2).

body_pred(atom, 1).
body_pred(reverse, 2).
body_pred(length, 2).
body_pred(delete, 3).
body_pred(nth0, 3).
body_pred(member, 2).


type(valid, (term_t,)).

%type(red, (color_t,)).
%type(green, (color_t,)).
%type(blue, (color_t,)).
%type(cyan, (color_t,)).
%type(magenta, (color_t,)).
%type(yellow, (color_t,)).
%type(small, (size_t,)).
%type(large, (size_t,)).
%type(triangle, (shape_t,)).
%type(circle, (shape_t,)).
%type(square, (shape_t,)).

%type(random, (comp_op,)).
%type(stack, (comp_op,)).
%type(grid, (comp_op,)).
%type(stack_reduce_bb, (comp_op,)).
%type(side_by_side, (comp_op,)).
%type(side_by_side_reduce_bb, (comp_op,)).
%type(diag_ul_lr, (comp_op,)).
%type(diag_ll_ur, (comp_op,)).

%type(quadrant_ul, (comp_op,)).
%type(quadrant_ur, (comp_op,)).
%type(quadrant_ll, (comp_op,)).
%type(quadrant_lr, (comp_op,)).


%type(quadrant, (comp_op,)).
%type(quadrant_or_center, (comp_op,)).

%type(color, (color_t,)).
%type(shape, (shape_t,)).
%type(size, (size_t,)).

%type(non_diag, (comp_op,)).
%type(diag, (comp_op,)).

%type(non_random, (comp_op,)).
%type(any_composition, (comp_op,)).
%type(line, (comp_op,)).

type(extract_shape, (term_t, shape_t)).
type(extract_color, (term_t, color_t)).
type(extract_size, (term_t, size_t)).

type(exists_shape, (shape_t, list_t)).
type(exists_color, (color_t, list_t)).
type(exists_size, (size_t, list_t)).

type(same_shape, (shape_t, list_t)).
type(same_color, (color_t, list_t)).
type(same_size, (size_t, list_t)).

type(contains, (term_t, term_t)).
type(recursive_contains, (term_t, term_t)).
type(contains_composition, (term_t, comp_op)).
type(contains_composition_depth, (term_t, comp_op, int)).

type(extract_operator, (term_t, comp_op)).
type(extract_children, (term_t, list_t)).
type(extract_op_and_chld, (term_t, comp_op, list_t)).

type(same_attribute, (list_t,)).
type(same_non_size, (list_t,)).
type(all_same, (list_t,)).

type(first, (list_t, term_t)).
type(last, (list_t, term_t)).
type(prepend, (term_t, list_t, list_t)).
type(droplast, (list_t, list_t)).
type(middle, (list_t, list_t)).
type(getmiddle, (list_t, term_t)).
type(dropmiddle, (list_t, list_t)).

type(less, (int, int)).
type(less_eq, (int, int)).
type(greater, (int, int)).
type(same_int, (int, int)).
type(different_int, (int, int)).

type(same_obj, (term_t, term_t)).
type(different_obj, (term_t, term_t)).

type(atom, (term_t,)).
type(reverse, (list_t, list_t)).
type(length, (list_t, int)).
type(delete, (list_t, term_t, list_t)).
type(nth0, (int, list_t, term_t)).
type(member, (term_t, list_t)).

direction(valid, (in,)).

%direction(red, (out,)).
%direction(green, (out,)).
%direction(blue, (out,)).
%direction(cyan, (out,)).
%direction(magenta, (out,)).
%direction(yellow, (out,)).
%direction(small, (out,)).
%direction(large, (out,)).
%direction(triangle, (out,)).
%direction(circle, (out,)).
%direction(square, (out,)).

%direction(random, (out,)).
%direction(stack, (out,)).
%direction(grid, (out,)).
%direction(stack_reduce_bb, (out,)).
%direction(side_by_side, (out,)).
%direction(side_by_side_reduce_bb, (out,)).
%direction(diag_ul_lr, (out,)).
%direction(diag_ll_ur, (out,)).

%direction(quadrant_ul, (out,)).
%direction(quadrant_ur, (out,)).
%direction(quadrant_ll, (out,)).
%direction(quadrant_lr, (out,)).

%direction(quadrant, (out,)).
%direction(quadrant_or_center, (out,)).

%direction(color, (in,)).
%direction(shape, (in,)).
%direction(size, (in,)).

%direction(non_diag, (in,)).
%direction(diag, (in,)).

%direction(non_random, (in,)).
%direction(any_composition, (in,)).
%direction(line, (in,)).

direction(extract_shape, (in, out)).
direction(extract_color, (in, out)).
direction(extract_size, (in, out)).

direction(exists_shape, (out, in)).
direction(exists_color, (out, in)).
direction(exists_size, (out, in)).

direction(same_shape, (out, in)).
direction(same_color, (out, in)).
direction(same_size, (out, in)).

direction(contains, (in, out)).
direction(recursive_contains, (in, out)).
direction(contains_composition, (in, out)).
direction(contains_composition_depth, (in, out, out)).

direction(extract_operator, (in, out)).
direction(extract_children, (in, out)).
direction(extract_op_and_chld, (in, out, out)).

direction(same_attribute, (in,)).
direction(same_non_size, (in,)).
direction(all_same, (in,)).

direction(first, (in, out)).
direction(last, (in, out)).
direction(prepend, (in, in, out)).
direction(droplast, (in, out)).
direction(middle, (in, out)).
direction(getmiddle, (in, out)).
direction(dropmiddle, (in, out)).

direction(less, (in, in)).
direction(less_eq, (in, in)).
direction(greater, (in, in)).
direction(same_int, (in, in)).
direction(different_int, (in, in)).
direction(same_obj, (in, in)).
direction(different_obj, (in, in)).

direction(atom, (in,)).
direction(reverse, (in, out)).
direction(length, (in, out)).
direction(delete, (in, in, out)).
direction(nth0, (in, in, out)).
direction(member, (out, in)).
