% CHEAT PREDICATES FOR EASY CURRICULUM:

symmetric_list(L) :- reverse(L, L).

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

% CHEAT PREDICATES FOR HARD CURRICULUM:

forall_shared_shape(C, SH) :- forall(contains(C, C1), (contains(C1, C2), extract_shape(C2, SH))).
forall_shared_color(C, CO) :- forall(contains(C, C1), (contains(C1, C2), extract_color(C2, CO))).
forall_shared_named_obj(C, X) :- forall(contains(C, C1), (contains(C1, C2), is_named_object(C2, X))).

pseudo_palindrome([]).
pseudo_palindrome([_]).
pseudo_palindrome(L) :- middle(L,M),pseudo_palindrome(M),last(L,A),first(L,B), same_shape(_, [A,B]).
pseudo_palindrome(L) :- middle(L,M),pseudo_palindrome(M),last(L,A),first(L,B), same_color(_, [A,B]).

pseudo_palindrome2([]).
pseudo_palindrome2([_]).
pseudo_palindrome2(L) :- middle(L,M),pseudo_palindrome2(M),last(L,A),first(L,B), same_shape(_, [A,B]).
pseudo_palindrome2(L) :- middle(L,M),pseudo_palindrome2(M),last(L,A),first(L,B), same_color(_, [A,B]).
pseudo_palindrome2(L) :- middle(L,M),pseudo_palindrome2(M),last(L,C1),first(L,C2), is_named_object(C1, X), is_named_object(C2, X).


