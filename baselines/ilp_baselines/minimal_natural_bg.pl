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

extract_operator(C, COMP) :- functor(C, COMP, 1).
extract_children(C, L) :- functor(C, _, 1), arg(1, C, L).
extract_op_and_chld(C, COMP, L) :- functor(C, COMP, 1), arg(1, C, L).


% USEFUL BUILT-IN PREDICATES:
% atom(X)
% reverse(L1, L2)
% length(L, N)
