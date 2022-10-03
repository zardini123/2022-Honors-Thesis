#!/usr/bin/env python3
"""Umbilics."""

import sys

import sympy
import sympy.vector

def bernstein_polynomial(degree: int, index: int, parameter: float):
    # Args: n, k
    return sympy.binomial(degree, index) \
        * (parameter ** index) \
        * ((1 - parameter) ** (degree - index))

def symbolic_dot_3D(a,b):
    i_symbol = sympy.Symbol('i')
    out = sympy.Sum(
        a[i_symbol, 0] * b[i_symbol, 0],
        (i_symbol, 0, 3)
    )
    return out

def symbolic_cross_3D(a,b):
    return sympy.Matrix(
        a[1,0] * b[2,0] - a[2,0] * b[1,0],
        a[2,0] * b[0,0] - a[0,0] * b[2,0],
        a[0,0] * b[1,0] - a[1,0] * b[0,0]
    )

def main() -> int:

    u_num_control_points = 4
    v_num_control_points = 4

    degree_u = u_num_control_points - 1
    degree_v = v_num_control_points - 1

    # Symbolic cross product example:
    #   https://github.com/sympy/sympy/blob/26f7bdbe3f860e7b4492e102edec2d6b429b5aaf/sympy/vector/vector.py#L471

    # Indexed Objects
    #   "represent a matrix element M[i, j]"
    #   https://docs.sympy.org/latest/modules/tensor/indexed.html
    points_symbols = sympy.IndexedBase('P')

    # Defining symbols in sympy:
    #   https://stackoverflow.com/a/14778490/6183001
    u_symbol, v_symbol = sympy.symbols('u v')
    sym_degree_u, sym_degree_v = sympy.symbols('n m')

    i_symbol, j_symbol = sympy.symbols('i j')

    # Sum documentation:
    #   Second argument: (dummy_variable, start, end)
    #   https://docs.sympy.org/latest/modules/concrete.html#sympy.concrete.summations.Sum
    # Surface equation: Bézier surface
    #   https://en.wikipedia.org/wiki/Bézier_surface
    surface = sympy.Sum(
        sympy.Sum(
            bernstein_polynomial(sym_degree_u, i_symbol, u_symbol)
            * bernstein_polynomial(sym_degree_v, j_symbol, v_symbol)
            * points_symbols[j_symbol, i_symbol],
            (j_symbol, 0, sym_degree_v)
        ),
        (i_symbol, 0, sym_degree_u)
    )

    # surface = sympy.Symbol('s')

    print("Unevaluated surface:")
    print(surface)

    # Substitute degrees
    #   Subs documentation:
    #       https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs
    surface = surface.subs({
        sym_degree_u: degree_u,
        sym_degree_v: degree_v,
    })

    # Evaluate summations
    surface = surface.doit()

    # Express points under a coordinate system:
    #   https://docs.sympy.org/latest/modules/vector/basics.html#points
    # coordinate_system = sympy.vector.CoordSys3D('N')

    # Initalize 2d array:
    #   https://stackoverflow.com/a/44382900/6183001
    points_2d_array = [[None] * u_num_control_points for _ in range(v_num_control_points)]

    for i in range(u_num_control_points):
        for j in range(v_num_control_points):

            point = sympy.MatrixSymbol(f'P[{i}, {j}]', 3, 1)

            # a,b,c = sympy.symbols('a b c')
            # point = coordinate_system.origin.locate_new(f'aaaa{i},{j}', a*coordinate_system.i + b*coordinate_system.j + c*coordinate_system.k)

            # "How to correctly substitude indexed variable in sympy?"
            #   https://stackoverflow.com/q/51403825/6183001
            surface = surface.subs({
                points_symbols[i, j]: point
            })

            # print(points_2d_array[j][i])
            # print(type(points_2d_array[j][i]))
            # print(points_2d_array[j][i].express_coordinates(coordinate_system))
            # print(points_2d_array[j][i].position_wrt(coordinate_system.origin))

    print(surface)

    # Preliminary for First fundemental magnitudes
    u_derivative = sympy.diff(surface, u_symbol)
    v_derivative = sympy.diff(surface, v_symbol)

    print("u_derivative:\t", u_derivative)

    # First fundemental magnitudes
    E = sympy.vector.Dot(u_derivative, u_derivative)
    F = sympy.vector.Dot(u_derivative, v_derivative)
    G = sympy.vector.Dot(v_derivative, v_derivative)

    print("E:\t", E)
    print("F:\t", F)
    print("G:\t", G)

    # Preliminary for Second fundemental magnitudes
    normal_vector = sympy.vector.Cross(u_derivative, v_derivative)
    unit_normal_vector = normal_vector.normalize()

    u_u_derivative = sympy.diff(surface, u_symbol, u_symbol)
    u_v_derivative = sympy.diff(surface, u_symbol, v_symbol)
    v_v_derivative = sympy.diff(surface, v_symbol, v_symbol)

    # Second fundemental magnitudes
    L = unit_normal_vector.dot(u_u_derivative)
    M = unit_normal_vector.dot(u_v_derivative)
    N = unit_normal_vector.dot(v_v_derivative)

    print(L)
    print(M)
    print(N)

    return 0

if __name__ == '__main__':
    sys.exit(main())
