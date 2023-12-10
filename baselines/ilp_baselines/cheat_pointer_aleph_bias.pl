:- modeb(*, symmetric_list(+list_t)).

:- modeb(*, is_named_object(+term_t, #named_obj_t)).
:- modeb(*, forall_shared_shape(+term_t, #shape_t)).
:- modeb(*, forall_shared_color(+term_t, #color_t)).
:- modeb(*, forall_shared_named_obj(+term_t, #named_obj_t)).

:- modeb(*, pseudo_palindrome(+list_t)).
:- modeb(*, pseudo_palindrome2(+list_t)).


:- determination(valid/1, symmetric_list/1).

:- determination(valid/1, is_named_object/2).
:- determination(valid/1, forall_shared_shape/2).
:- determination(valid/1, forall_shared_color/2).
:- determination(valid/1, forall_shared_named_obj/2).
:- determination(valid/1, pseudo_palindrome/1).
:- determination(valid/1, pseudo_palindrome2/1).
