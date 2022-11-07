#!/usr/bin/env python3

import sys
import inspect

import sympy
import sympy.vector

import shared_math


def main() -> int:

    u_symbol, v_symbol = sympy.symbols('u v')
    sym_degree_u, sym_degree_v = sympy.symbols('n m')

    # Indexed Objects
    #   "represent a matrix element M[i, j]"
    #   https://docs.sympy.org/latest/modules/tensor/indexed.html
    points_symbols = sympy.IndexedBase('P')

    i_symbol, j_symbol = sympy.symbols('i j')

    # Sum documentation:
    #   Second argument: (dummy_variable, start, end)
    #   https://docs.sympy.org/latest/modules/concrete.html#sympy.concrete.summations.Sum
    surface = sympy.Sum(
        sympy.Sum(
            shared_math.bernstein_polynomial(sym_degree_u, i_symbol, u_symbol)
            * shared_math.bernstein_polynomial(sym_degree_v, j_symbol, v_symbol)
            * points_symbols[j_symbol, i_symbol],
            (j_symbol, 0, sym_degree_v)
        ),
        (i_symbol, 0, sym_degree_u)
    )

    surface_lambda = sympy.lambdify([u_symbol, v_symbol], surface, modules=["numpy", "math"])
    lines = inspect.getsource(surface_lambda)
    print(lines)

    # print("Unevaluated surface:")
    # print(surface)

    # surface = surface.subs({
    #     bernstein_polynomial_func: bernstein_polynomial
    # })
    #
    # print(surface)

    # return 0

    # Substitute degrees
    #   Subs documentation:
    #       https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs
    # surface = surface.subs({
    #     sym_degree_u: degree_u,
    #     sym_degree_v: degree_v,
    # })

    # Evaluate summations
    # surface = surface.doit()

    return 0


if __name__ == '__main__':
    sys.exit(main())
