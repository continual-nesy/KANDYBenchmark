:- modeh(*, valid(+term_t)).

%:- determination(valid/1, red/1).
%:- determination(valid/1, green/1).
%:- determination(valid/1, blue/1).
%:- determination(valid/1, cyan/1).
%:- determination(valid/1, magenta/1).
%:- determination(valid/1, yellow/1).
%:- determination(valid/1, small/1).
%:- determination(valid/1, large/1).
%:- determination(valid/1, triangle/1).
%:- determination(valid/1, circle/1).
%:- determination(valid/1, square/1).

%:- determination(valid/1, random/1).
%:- determination(valid/1, stack/1).
%:- determination(valid/1, grid/1).
%:- determination(valid/1, stack_reduce_bb/1).
%:- determination(valid/1, side_by_side/1).
%:- determination(valid/1, side_by_side_reduce_bb/1).
%:- determination(valid/1, diag_ul_lr/1).
%:- determination(valid/1, diag_ll_ur/1).


%:- determination(valid/1, color/1).
%:- determination(valid/1, shape/1).
%:- determination(valid/1, size/1).

%:- determination(valid/1, non_diag/1).
%:- determination(valid/1, diag/1).

%:- determination(valid/1, non_random/1).
%:- determination(valid/1, any_composition/1).
%:- determination(valid/1, line/1).

%:- determination(valid/1, quadrant/1).
%:- determination(valid/1, quadrant_or_center/1).

%:- determination(valid/1, quadrant_ul/1).
%:- determination(valid/1, quadrant_ur/1).
%:- determination(valid/1, quadrant_ll/1).
%:- determination(valid/1, quadrant_lr/1).

:- determination(valid/1, extract_shape/2).
:- determination(valid/1, extract_color/2).
:- determination(valid/1, extract_size/2).

:- determination(valid/1, exists_shape/2).
:- determination(valid/1, exists_color/2).
:- determination(valid/1, exists_size/2).

:- determination(valid/1, same_shape/2).
:- determination(valid/1, same_color/2).
:- determination(valid/1, same_size/2).

:- determination(valid/1, contains/2).
:- determination(valid/1, recursive_contains/2).
:- determination(valid/1, contains_composition/2).
:- determination(valid/1, contains_composition_depth/3).

:- determination(valid/1, extract_operator/2).
:- determination(valid/1, extract_children/2).
:- determination(valid/1, extract_op_and_chld/3).

:- determination(valid/1, same_attribute/1).
:- determination(valid/1, same_non_size/1).
:- determination(valid/1, all_same/1).

:- determination(valid/1, first/2).
:- determination(valid/1, last/2).
:- determination(valid/1, prepend/3).
:- determination(valid/1, droplast/2).
:- determination(valid/1, middle/2).
:- determination(valid/1, getmiddle/2).
:- determination(valid/1, dropmiddle/2).

:- determination(valid/1, less/2).
:- determination(valid/1, less_eq/2).
:- determination(valid/1, greater/2).
:- determination(valid/1, same_int/2).
:- determination(valid/1, different_int/2).
:- determination(valid/1, same_obj/2).
:- determination(valid/1, different_obj/2).

:- determination(valid/1, atom/1).
:- determination(valid/1, reverse/2).
:- determination(valid/1, length/2).
:- determination(valid/1, delete/3).
:- determination(valid/1, nth0/3).
:- determination(valid/1, member/2).


%:- modeb(*, red(#color_t)).
%:- modeb(*, green(#color_t)).
%:- modeb(*, blue(#color_t)).
%:- modeb(*, cyan(#color_t)).
%:- modeb(*, magenta(#color_t)).
%:- modeb(*, yellow(#color_t)).
%:- modeb(*, small(#size_t)).
%:- modeb(*, large(#size_t)).
%:- modeb(*, triangle(#shape_t)).
%:- modeb(*, circle(#shape_t)).
%:- modeb(*, square(#shape_t)).

%:- modeb(*, random(#comp_op)).
%:- modeb(*, stack(#comp_op)).
%:- modeb(*, grid(#comp_op)).
%:- modeb(*, stack_reduce_bb(#comp_op)).
%:- modeb(*, side_by_side(#comp_op)).
%:- modeb(*, side_by_side_reduce_bb(#comp_op)).
%:- modeb(*, diag_ul_lr(#comp_op)).
%:- modeb(*, diag_ll_ur(#comp_op)).

%:- modeb(*, color(#color_t)).
%:- modeb(*, shape(#shape_t)).
%:- modeb(*, size(#size_t)).

%:- modeb(*, non_diag(#comp_op)).
%:- modeb(*, diag(#comp_op)).

%:- modeb(*, non_random(#comp_op)).
%:- modeb(*, any_composition(#comp_op)).
%:- modeb(*, line(#comp_op)).

%:- modeb(*, quadrant(#comp_op)).
%:- modeb(*, quadrant_or_center(#comp_op)).

%:- modeb(*, quadrant_ul(#comp_op)).
%:- modeb(*, quadrant_ur(#comp_op)).
%:- modeb(*, quadrant_ll(#comp_op)).
%:- modeb(*, quadrant_lr(#comp_op)).

:- modeb(*, extract_shape(+term_t, #shape_t)).
:- modeb(*, extract_color(+term_t, #color_t)).
:- modeb(*, extract_size(+term_t, #size_t)).

:- modeb(*, exists_shape(#shape_t, +list_t)).
:- modeb(*, exists_color(#color_t, +list_t)).
:- modeb(*, exists_size(#size_t, +list_t)).

:- modeb(*, same_shape(#shape_t, +list_t)).
:- modeb(*, same_color(#color_t, +list_t)).
:- modeb(*, same_size(#size_t, +list_t)).

:- modeb(*, contains(+term_t, -term_t)).
:- modeb(*, recursive_contains(+term_t, -term_t)).
:- modeb(*, contains_composition(+term_t, #comp_op)).
:- modeb(*, contains_composition_depth(+term_t, #comp_op, -int)).

:- modeb(*, extract_operator(+term_t, #comp_op)).
:- modeb(*, extract_children(+term_t, -list_t)).
:- modeb(*, extract_op_and_chld(+term_t, #comp_op, -list_t)).

:- modeb(*, same_attribute(+list_t)).
:- modeb(*, same_non_size(+list_t)).
:- modeb(*, all_same(+list_t)).

:- modeb(*, first(+list_t, -term_t)).
:- modeb(*, last(+list_t, -term_t)).
:- modeb(*, prepend(+, +list_t, -list_t)).
:- modeb(*, droplast(+list_t, -list_t)).
:- modeb(*, middle(+list_t, -list_t)).
:- modeb(*, getmiddle(+list_t, -term_t)).
:- modeb(*, dropmiddle(+list_t, -list_t)).

:- modeb(*, less(+int, +int)).
:- modeb(*, less_eq(+int, +int)).
:- modeb(*, greater(+int, +int)).
:- modeb(*, same_int(+int, +int)).
:- modeb(*, different_int(+int, +int)).
:- modeb(*, same_obj(+term_t, +term_t)).
:- modeb(*, different_obj(+term_t, +term_t)).

:- modeb(*, atom(+term_t)).
:- modeb(*, reverse(+list_t, -list_t)).
:- modeb(*, length(+list_t, -int)).
:- modeb(*, delete(+list_t, +term_t, -list_t)).
:- modeb(*, nth0(+int, +list_t, -term_t)).
:- modeb(*, member(-term_t, +)).

