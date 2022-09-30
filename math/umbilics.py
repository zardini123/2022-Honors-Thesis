#!/usr/bin/env python3
"""Umbilics."""

import sys

import sympy

def bernstein_polynomial(degree: int, index: int, parameter: float):
    # Args: n, k
    return sympy.binomial(degree, index) * (parameter ** index) * ((1 - parameter) ** (degree - index))

def main() -> int:

    u_control_points = 4
    v_control_points = 4

    degree_u = u_control_points - 1
    degree_v = v_control_points - 1

    # Defining symbols in sympy:
    #   https://stackoverflow.com/a/14778490/6183001
    sym_u, sym_v = sympy.symbols('u v')
    sym_degree_u, sym_degree_v = sympy.symbols('n m')

    i, j = sympy.symbols('i j')

    # Second argument: (dummy_variable, start, end)
    # Sum documentation:
    #   https://docs.sympy.org/latest/modules/concrete.html#sympy.concrete.summations.Sum
    bezier_surface = sympy.Sum(
        sympy.Sum(
            bernstein_polynomial(sym_degree_u, i, sym_u) * bernstein_polynomial(sym_degree_v, j, sym_v),
            (j, 0, sym_degree_v)
        ),
        (i, 0, sym_degree_u)
    )
    print(bezier_surface)

    # Subs documentation:
    #   https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs
    bezier_surface = bezier_surface.subs({
        sym_degree_u: degree_u,
        sym_degree_v: degree_v,
    })

    print(bezier_surface)

    # d_f_with_x = sympy.diff(f, x)

    # print(d_f_with_x)

    return 0

if __name__ == '__main__':
    sys.exit(main())
