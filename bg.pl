color(red).
color(green).
color(blue).
color(cyan).
color(magenta).
color(yellow).

shape(triangle).
shape(circle).
shape(square).

size(small).
size(large).

non_diag(stack).
non_diag(side_by_side).
non_diag(stack_reduce_bb).
non_diag(side_by_side_reduce_bb).
non_diag(grid).

diag(diag_ul_lr).
diag(diag_ll_ur).

quadrant_ul(quadrant_ul).
quadrant_ur(quadrant_ur).
quadrant_ll(quadrant_ll).
quadrant_lr(quadrant_lr).

quadrant(quadrant_ul).
quadrant(quadrant_ur).
quadrant(quadrant_ll).
quadrant(quadrant_lr).

quadrant_or_center(in).
quadrant_or_center(X) :- quadrant(X).

non_random(X) :- diag(X).
non_random(X) :- non_diag(X).
non_random(X) :- quadrant(X).
any_composition(X) :- non_random(X).
any_composition(random).


line(X) :- non_diag(X), X \= grid.
line(X) :- diag(X).

house(C) :- extract_op_and_chld(C, stack, [C1, C2]), extract_shape(C1, triangle), extract_shape(C2, square), same_size(_, [C1, C2]).
car(C) :- extract_op_and_chld(C, side_by_side, [C1, C2]), extract_shape(C1, circle), extract_shape(C2, circle), same_size(_, [C1, C2]), same_color(_, [C1, C2]).
tower(C) :- extract_op_and_chld(C, stack, L), same_shape(square, L), same_size(_, L), length(L, N), N >= 2, N =< 3.
wagon(C) :- extract_op_and_chld(C, side_by_side, L), same_shape(square, L), same_size(_, L), length(L, N), N >= 2, N =< 3.
traffic_light(C) :- extract_op_and_chld(C, stack, [C1, C2, C3]), same_shape(circle, [C1, C2, C3]), same_size(_, [C1, C2, C3]), extract_color(C1, red), extract_color(C2, yellow), extract_color(C3, green).
named_object(house).
named_object(car).
named_object(tower).
named_object(wagon).
named_object(traffic_light).
is_named_object(C, house) :- house(C).
is_named_object(C, car) :- car(C).
is_named_object(C, tower) :- tower(C).
is_named_object(C, wagon) :- wagon(C).
is_named_object(C, traffic_light) :- traffic_light(C).

shape_props(T, SH, CO, SZ) :- atom(T), term_string(T, S), split_string(S, "_", "", L), L = [SH, CO, SZ].
extract_shape(T, SH) :- shape_props(T, SH1, _, _), term_string(SH, SH1), shape(SH).
extract_color(T, CO) :- shape_props(T, _, CO1, _), term_string(CO, CO1), color(CO).
extract_size(T, SZ) :- shape_props(T, _, _, SZ1), term_string(SZ, SZ1), size(SZ).

exists_shape(SH, [H|_]) :- extract_shape(H, SH).
exists_shape(SH, [_|T]) :- exists_shape(SH, T).

same_shape(SH, [H]) :- extract_shape(H, SH).
same_shape(SH, [H|T]) :- extract_shape(H, SH), same_shape(SH, T).

exists_color(CO, [H|_]) :- extract_color(H, CO).
exists_color(CO, [_|T]) :- exists_color(CO, T).

same_color(CO, [H]) :- extract_color(H, CO).
same_color(CO, [H|T]) :- extract_color(H, CO), same_color(CO, T).

exists_size(SZ, [H|_]) :- extract_size(H, SZ).
exists_size(SZ, [_|T]) :- exists_size(SZ, T).

same_size(SZ, [H]) :- extract_size(H, SZ).
same_size(SZ, [H|T]) :- extract_size(H, SZ), same_size(SZ, T).

contains(C, X) :- extract_children(C, L), member(X, L).

recursive_contains(C, X) :- contains(C, X), atom(X).
recursive_contains(C, X) :- contains(C, C1), recursive_contains(C1, X).


recursive_contains2(C, X, 0) :- contains(C, X), functor(C, _, 1).
recursive_contains2(C, X, I) :- contains(C, C1), recursive_contains2(C1, X, J), I is J + 1.

contains_composition(C, COMP) :- extract_operator(C, COMP).
contains_composition(C, COMP) :- recursive_contains2(C, C1, _), extract_operator(C1, COMP).

contains_composition_depth(C, COMP, 0) :- extract_operator(C, COMP).
contains_composition_depth(C, COMP, I) :- recursive_contains2(C, C1, J), extract_operator(C1, COMP), I is J + 1.

extract_operator(C, COMP) :- functor(C, COMP, 1).
extract_children(C, L) :- functor(C, _, 1), arg(1, C, L).
extract_op_and_chld(C, COMP, L) :- functor(C, COMP, 1), arg(1, C, L).

same_attribute(L) :- same_shape(_, L).
same_attribute(L) :- same_color(_, L).
same_attribute(L) :- same_size(_, L).

same_non_size(L) :- same_shape(_, L).
same_non_size(L) :- same_color(_, L).

all_same(H, [H]).
all_same(H, [H|T]) :- all_same(H, T).


expand2([A, B], A, B).
expand4([A, B, C, D], A, B, C, D).
expand8([A, B, C, D, E, F, G, H], A, B, C, D, E, F, G, H).
expand9([A, B, C, D, E, F, G, H, I], A, B, C, D, E, F, G, H, I).
odd(N) :- N mod 2 =:= 1.
even(N) :- N mod 2 =:= 0.


first([H|_],H).

last([H], H).
last([_|T],X):- last(T, X).


prepend(X, L, [X|L]).
droplast([_], []).
droplast([H|T], [H|T2]):- droplast(T, T2).

middle([_|T], T2):- droplast(T, T2).
getmiddle(L, X) :- length(L, N), odd(N), N1 is div(N, 2), nth0(N1, L, X).
dropmiddle(L, L1) :- getmiddle(L, X), delete(L, X, L1).


less_eq(N, N1) :- N =< N1.
less(N, N1) :- N < N1.
greater(N, N1) :- N > N1.

same(X, Y) :- X = Y.
different(X, Y) :- X \= Y.

% USEFUL BUILT-IN PREDICATES:
% atom(X)
% reverse(L1, L2)
% length(L, N)
% delete(L, X, L1)
% nth0(N, L, X)
% member(X, L)
