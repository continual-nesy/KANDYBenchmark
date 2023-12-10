body_pred(symmetric_list, 1).

%body_pred(house, 1).
%body_pred(car, 1).
%body_pred(tower, 1).
%body_pred(wagon, 1).
%body_pred(traffic_light, 1).
%body_pred(named_object, 1).
body_pred(is_named_object, 2).
body_pred(forall_shared_shape, 2).
body_pred(forall_shared_color, 2).
body_pred(forall_shared_named_obj, 2).
body_pred(pseudo_palindrome, 1).
body_pred(pseudo_palindrome2, 1).

type(symmetric_list, (list_t,)).

%type(house, (term_t,)).
%type(car, (term_t,)).
%type(tower, (term_t,)).
%type(wagon, (term_t,)).
%type(traffic_light, (term_t,)).
%type(named_object, (named_obj_t,)).
type(is_named_object, (term_t, named_obj_t)).
type(forall_shared_shape, (term_t, shape_t)).
type(forall_shared_color, (term_t, color_t)).
type(forall_shared_named_obj, (term_t, named_object_t)).

type(pseudo_palindrome, (list_t,)).
type(pseudo_palindrome2, (list_t,)).

direction(symmetric_list, (in,)).

%direction(house, (in,)).
%direction(car, (in,)).
%direction(tower, (in,)).
%direction(wagon, (in,)).
%direction(traffic_light, (in,)).
%direction(named_object, (in,)).
direction(is_named_object, (in, out)).
direction(forall_shared_shape, (in, out)).
direction(forall_shared_color, (in, out)).
direction(forall_shared_named_obj, (in, out)).

direction(pseudo_palindrome, (in,)).
direction(pseudo_palindrome2, (in,)).

