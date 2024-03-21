color(red).
color(green).
color(blue).
color(cyan).
color(magenta).
color(yellow).

shape(triangle).
shape(downtriangle).
shape(diamond).
shape(square).
shape(circle).
shape(ellipsev).
shape(ellipseh).

size(small).
size(medium).
size(large).

primary_color(red).
primary_color(green).
primary_color(blue).
secondary_color(C) :- color(C), not(primary_color(C)).

round_shape(circle).
round_shape(ellipseh).
round_shape(ellipsev).
polygon_shape(S) :- shape(S), not(round_shape(S)).

three_side(triangle).
three_side(downtriangle).
four_side(square).
four_side(diamond).


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

hierarchical_same_color(CO, [H]) :- atom(H), extract_color(H, CO).
hierarchical_same_color(CO, [H]) :- extract_children(H, L), hierarchical_same_color(CO, L).
hierarchical_same_color(CO, [H|T]) :- atom(H), extract_color(H, CO), hierarchical_same_color(CO, T).
hierarchical_same_color(CO, [H|T]) :- extract_children(H, L), hierarchical_same_color(CO, L), hierarchical_same_color(CO, T).

hierarchical_same_size(SZ, [H]) :- atom(H), extract_size(H, SZ).
hierarchical_same_size(SZ, [H]) :- extract_children(H, L), hierarchical_same_size(SZ, L).
hierarchical_same_size(SZ, [H|T]) :- atom(H), extract_size(H, SZ), hierarchical_same_size(SZ, T).
hierarchical_same_size(SZ, [H|T]) :- extract_children(H, L), hierarchical_same_size(SZ, L), hierarchical_same_size(SZ, T).

hierarchical_object(C, COL, SZ) :- extract_children(C, L), hierarchical_same_color(COL, L), hierarchical_same_size(SZ, L).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2-level hierarchies

% stack: triangle, square
obj1(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, [C1, C2]), member(OP, [stack, stack_reduce_bb]), extract_shape(C1, triangle), extract_shape(C2, square).

% horizontal sequence of squares
obj2(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [side_by_side, side_by_side_reduce_bb]), same_shape(square, L), length(L, N), N >= 2, N =< 5.

% vertical sequence of squares
obj3(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [stack, stack_reduce_bb]), same_shape(square, L), length(L, N), N >= 2, N =< 5.

% diagonal sequence of diamonds
obj4(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [diag_ul_lr, diag_ll_ur]), same_shape(diamond, L), length(L, N), N >= 2, N =< 5.

% stack: square down triangle
obj5(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, [C1, C2]), member(OP, [stack, stack_reduce_bb]), extract_shape(C1, square), extract_shape(C2, downtriangle).

% stack: triangle, down triangle
obj6(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, [C1, C2]), member(OP, [stack, stack_reduce_bb]), extract_shape(C1, triangle), extract_shape(C2, downtriangle).

% sequence of circles
obj7(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [side_by_side, side_by_side_reduce_bb, stack, stack_reduce_bb, diag_ul_lr, diag_ll_ur]), same_shape(circle, L), length(L, N), N >= 2, N =< 5.

% horizontal sequence of ellipses (same direction)
obj8(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [side_by_side_reduce_bb, side_by_side]), same_shape(SH, L), member(SH, [ellipseh, ellipsev]), length(L, N), N >= 2, N =< 5.

% stack: ellipse_v, circle, ellipse_h
%obj9(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, [C1, C2, C3]), member(OP, [stack, stack_reduce_bb]), extract_shape(C1, ellipse_v), extract_shape(C2, circle), extract_shape(C3, ellipse_h).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
same_object(O, [O]).
same_object(O, [O|T]) :- same_object(O, T).

halve_list(L, A, B) :- length(L, N), H is N // 2, length(A, H), append(A, [_|B], L).

% 3-level hierarchies
% horizontal sequence of squares, the central element is an obj1
obj10(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [side_by_side, side_by_side_reduce_bb]), length(L, N), N >= 2, N =< 5, N mod 2 =:= 1, getmiddle(L, O1), dropmiddle(L, L1), obj1(O1), same_shape(square, L1).

% horizontal sequence of obj1
obj11(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [side_by_side, side_by_side_reduce_bb]), length(L, N), N >= 2, N =< 5, same_object(O, L), obj1(O).

% cross of squares (combination of obj2 and obj3)
obj12(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, side_by_side, L), length(L, N), N >= 2, N =< 5, N mod 2 =:= 1, getmiddle(L, C1), extract_op_and_chld(C1, stack, L1), length(L1, N), same_shape(square, L1), dropmiddle(L, L2), same_shape(square, L2).

% diagonal cross of diamonds (combination of 2 obj4) NO!
%obj13(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, diag_ul_lr, L), length(L, N), N >= 2, N =< 5, N mod 2 =:= 1, getmiddle(L, C1), extract_op_and_chld(C1, diag_ll_ur, L1), length(L1, N), same_shape(diamond, L1), dropmiddle(L, L2), same_shape(diamond, L2).

% horizontal sequence of obj1, obj5 and obj6 such as obj6 appears only once in the middle, and obj1 appears only on the left (right respectively) side while obj5 appears only on the right (left respectively) side of obj6.
obj14(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [side_by_side, side_by_side_reduce_bb]), length(L, N), N >= 2, N =< 5, N mod 2 =:= 1, getmiddle(L, C1), obj6(C1), halve_list(L, L1, L2), same_object(O1, L1), same_object(O2, L2), obj1(O1), obj5(O2).
obj14(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, OP, L), member(OP, [side_by_side, side_by_side_reduce_bb]), length(L, N), N >= 2, N =< 5, N mod 2 =:= 1, getmiddle(L, C1), obj6(C1), halve_list(L, L1, L2), same_object(O1, L1), same_object(O2, L2), obj5(O1), obj1(O2).

% cross/diagonal cross of circles
obj15(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, side_by_side, L), length(L, N), N >= 2, N =< 5, N mod 2 =:= 1, getmiddle(L, C1), extract_op_and_chld(C1, stack, L1), length(L1, N), same_shape(circle, L1), dropmiddle(L, L2), same_shape(circle, L2).
%obj15(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, diag_ul_lr, L), length(L, N), N >= 2, N =< 5, N mod 2 =:= 1, getmiddle(L, C1), extract_op_and_chld(C1, diag_ll_ur, L1), length(L1, N), same_shape(circle, L1), dropmiddle(L, L2), same_shape(circle, L2).

% T of circles
obj16(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, stack, [C1|L]), extract_op_and_chld(C1, side_by_side, L1), same_shape(circle, L), same_shape(circle, L1).

% inverted T of ellipses
obj17(C) :- hierarchical_object(C, _, _), extract_op_and_chld(C, stack, L), droplast(L, L1), last(L, C1), extract_op_and_chld(C1, side_by_side, L2), same_shape(ellipsev, L1), same_shape(ellipseh, L2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


first([H|_],H).

last([H], H).
last([_|T],X):- last(T, X).


prepend(X, L, [X|L]).
droplast([_], []).
droplast([H|T], [H|T2]):- droplast(T, T2).

middle([_|T], T2):- droplast(T, T2).
getmiddle(L, X) :- length(L, N), N mod 2 =:= 1, N1 is div(N, 2), nth0(N1, L, X).
dropmiddle(L, L1) :- getmiddle(L, X), delete(L, X, L1).

same(X, Y) :- X = Y.
different(X, Y) :- X \= Y.

% USEFUL BUILT-IN PREDICATES:
% atom(X)
% reverse(L1, L2)
% length(L, N)
% delete(L, X, L1)
% nth0(N, L, X)
% member(X, L)