#!/usr/bin/env python3
"""Umbilics."""

import sys

import sympy
import sympy.vector

# Sympy docs: Creating a custom function:
#   https://docs.sympy.org/latest/guides/custom-functions.html#creating-a-custom-function
# class BernsteinPolynomial(sympy.Function):
#     @classmethod
#     def eval(cls, degree, index, parameter):
#         # Args: n, k
#         return sympy.binomial(degree, index) \
#             * (parameter ** index) \
#             * ((1 - parameter) ** (degree - index))


def bernstein_polynomial(degree, index, parameter):
    # Args: n, k
    return sympy.binomial(degree, index) \
        * (parameter ** index) \
        * ((1 - parameter) ** (degree - index))

def main() -> int:

    u_num_control_points = 3
    v_num_control_points = 3

    # Symbolic cross product example:
    #   https://github.com/sympy/sympy/blob/26f7bdbe3f860e7b4492e102edec2d6b429b5aaf/sympy/vector/vector.py#L471

    # Express points under a coordinate system:
    #   https://docs.sympy.org/latest/modules/vector/basics.html#points
    coord_sys = sympy.vector.CoordSys3D('C')

    # Creates 3 symbols, x, y, and z for each point
    point_component_scalar_symbols = [
        [
            sympy.symbols(f'P[{i}\,{j}]_x:z')
            for i in range(u_num_control_points)
        ] for j in range(v_num_control_points)
    ]

    # print(point_component_scalar_symbols)

    points = [
        [
            (coord_sys.i * point_component_scalar_symbols[j][i][0]
            + coord_sys.j * point_component_scalar_symbols[j][i][1]
            + coord_sys.k * point_component_scalar_symbols[j][i][2])
            for i in range(u_num_control_points)
        ] for j in range(v_num_control_points)
    ]

    # print(points)

    degree_u = u_num_control_points - 1
    degree_v = v_num_control_points - 1

    # Surface equation preliminaries
    u_symbol, v_symbol = sympy.symbols('u v')
    sym_degree_u, sym_degree_v = sympy.symbols('n m')

    i_symbol, j_symbol = sympy.symbols('i j')

    # Sum documentation:
    #   Second argument: (dummy_variable, start, end)
    #   https://docs.sympy.org/latest/modules/concrete.html#sympy.concrete.summations.Sum
    # surface = sympy.Sum(
    #     sympy.Sum(
    #         bernstein_polynomial(sym_degree_u, i_symbol, u_symbol)
    #         * bernstein_polynomial(sym_degree_v, j_symbol, v_symbol)
    #         * points[j_symbol][i_symbol],
    #         (j_symbol, 0, sym_degree_v)
    #     ),
    #     (i_symbol, 0, sym_degree_u)
    # )

    bernstein_polynomial_func = sympy.Function('B', real=True)

    # Surface equation: Bézier surface
    #   https://en.wikipedia.org/wiki/Bézier_surface
    surface = None
    for i in range(degree_u):
        for j in range(degree_v):
            addition_expr = bernstein_polynomial_func(degree_u, i, u_symbol) \
                * bernstein_polynomial_func(degree_v, j, v_symbol) \
                * points[j][i]

            if surface is None:
                surface = addition_expr
            else:
                surface += addition_expr

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

    # First fundemental magnitudes
    E_symbol, F_symbol, G_symbol = sympy.symbols('E F G')
    # Second fundemental magnitudes
    L_symbol, M_symbol, N_symbol = sympy.symbols('L M N')

    umbilic = sympy.Eq((
        (E_symbol * N_symbol)
        + (G_symbol * L_symbol)
        - (2 * F_symbol * M_symbol)
    ) ** 2 - (
        4 * ((L_symbol * N_symbol) - M_symbol ** 2)
        * ((E_symbol * G_symbol) - F_symbol ** 2)
    ), 0)

    umbilic = sympy.simplify(umbilic)

    print(sympy.pretty(umbilic))

    # Preliminary for First fundemental magnitudes
    u_derivative = sympy.diff(surface, u_symbol)
    v_derivative = sympy.diff(surface, v_symbol)

    # print("u_derivative:\t", u_derivative)

    # First fundemental magnitudes
    E = u_derivative.dot(u_derivative)
    F = u_derivative.dot(v_derivative)
    G = v_derivative.dot(v_derivative)

    # print("E:\t", E)
    # print("F:\t", F)
    # print("G:\t", G)

    # return 0

    # Preliminary for Second fundemental magnitudes
    normal_vector = u_derivative.cross(v_derivative)
    unit_normal_vector = normal_vector.normalize()

    # print(unit_normal_vector)

    u_u_derivative = sympy.diff(surface, u_symbol, u_symbol)
    u_v_derivative = sympy.diff(surface, u_symbol, v_symbol)
    v_v_derivative = sympy.diff(surface, v_symbol, v_symbol)

    # Second fundemental magnitudes
    L = unit_normal_vector.dot(u_u_derivative)
    M = unit_normal_vector.dot(u_v_derivative)
    N = unit_normal_vector.dot(v_v_derivative)

    # print(L)
    # print(M)
    # print(N)

    # print(umbilic)

    # return 0

    # print("start simplify")

    # out = sympy.solve(umbilic, u_symbol, v_symbol)
    # umbilic = sympy.collect(umbilic, u_symbol)
    # umbilic = sympy.collect(umbilic, v_symbol)

    # print(umbilic)

    # print("start simplify")
    # print(sympy.simplify(L))

    return 0

if __name__ == '__main__':
    sys.exit(main())
