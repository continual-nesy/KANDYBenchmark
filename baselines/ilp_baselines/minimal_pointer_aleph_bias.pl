:- modeh(*, valid(+sample)).

:- determination(valid/1, sample_is/2).

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

:- determination(valid/1, defined_as/3).

:- determination(valid/1, reverse/2).
:- determination(valid/1, length/2).

:- modeb(*, sample_is(+sample, -term_t)).

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

:- modeb(*, defined_as(+term_t, #comp_op, -list_t)).

:- modeb(*, reverse(+list_t, -list_t)).
:- modeb(*, length(+list_t, -int)).
