% Atomic concepts
c_triangle(C) :- recursive_contains(C, C1), extract_shape(C1, triangle).
c_downtriangle(C) :- recursive_contains(C, C1), extract_shape(C1, downtriangle).
c_square(C) :- recursive_contains(C, C1), extract_shape(C1, square).
c_diamond(C) :- recursive_contains(C, C1), extract_shape(C1, diamond).
c_circle(C) :- recursive_contains(C, C1), extract_shape(C1, circle).
c_ellipsev(C) :- recursive_contains(C, C1), extract_shape(C1, ellipsev).
c_ellipseh(C) :- recursive_contains(C, C1), extract_shape(C1, ellipseh).

c_red(C) :- recursive_contains(C, C1), extract_color(C1, red).
c_green(C) :- recursive_contains(C, C1), extract_color(C1, green).
c_blue(C) :- recursive_contains(C, C1), extract_color(C1, blue).
c_cyan(C) :- recursive_contains(C, C1), extract_color(C1, cyan).
c_magenta(C) :- recursive_contains(C, C1), extract_color(C1, magenta).
c_yellow(C) :- recursive_contains(C, C1), extract_color(C1, yellow).

c_small(C) :- recursive_contains(C, C1), extract_size(C1, small).
c_medium(C) :- recursive_contains(C, C1), extract_size(C1, medium).
c_large(C) :- recursive_contains(C, C1), extract_size(C1, large).

% Derived atomic concepts
c_round(C) :- recursive_contains(C, C1), extract_shape(C1, SH), round_shape(SH).
c_three_side(C) :- recursive_contains(C, C1), extract_shape(C1, SH), three_side(SH).
c_four_side(C) :- recursive_contains(C, C1), extract_shape(C1, SH), four_side(SH).
c_primary_color(C) :- recursive_contains(C, C1), extract_color(C1, COL), primary_color(COL).

% Second-level concepts
c_obj1(C) :- recursive_contains(C, C1), obj1(C1).
c_obj2(C) :- recursive_contains(C, C1), obj2(C1).
c_obj3(C) :- recursive_contains(C, C1), obj3(C1).
c_obj4(C) :- recursive_contains(C, C1), obj4(C1).
c_obj5(C) :- recursive_contains(C, C1), obj5(C1).
c_obj6(C) :- recursive_contains(C, C1), obj6(C1).
c_obj7(C) :- recursive_contains(C, C1), obj7(C1).
c_obj8(C) :- recursive_contains(C, C1), obj8(C1).

% Third-level concepts
c_obj10(C) :- recursive_contains(C, C1), obj10(C1).
c_obj11(C) :- recursive_contains(C, C1), obj11(C1).
c_obj12(C) :- recursive_contains(C, C1), obj12(C1).
c_obj14(C) :- recursive_contains(C, C1), obj14(C1).
c_obj15(C) :- recursive_contains(C, C1), obj15(C1).
c_obj16(C) :- recursive_contains(C, C1), obj16(C1).
c_obj17(C) :- recursive_contains(C, C1), obj17(C1).
