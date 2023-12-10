head_pred(valid, 1).

body_pred(sample_is, 2).

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

body_pred(defined_as, 3).

body_pred(reverse, 2).
body_pred(length, 2).



type(valid, (sample_t,)).
type(sample_is, (sample_t, term_t)).

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

type(defined_as, (term_t, comp_op, list_t)).

type(reverse, (list_t, list_t)).
type(length, (list_t, int)).

direction(valid, (in,)).
direction(sample_is, (in, out)).

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

direction(defined_as, (in, out, out)).

direction(reverse, (in, out)).
direction(length, (in, out)).