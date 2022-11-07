#!/usr/bin/env python3
"""Umbilics."""

import sys
import inspect
import pathlib

import sympy
import sympy.vector

import shared_math

OUTPUT_FOLDER = pathlib.Path("./output")


def change_lambda_function_name(function_source: str, new_name: str):
    function_source_lines = function_source.splitlines()

    function_source_lines[0] = function_source_lines[0].replace(
        "_lambdifygenerated", new_name
    )

    new_source = '\n'.join(function_source_lines)

    return new_source


def get_function_source(lambda_func, function_name):
    umbilic_function_source = inspect.getsource(lambda_func)
    final_source = change_lambda_function_name(umbilic_function_source, function_name)

    return final_source


def convert_sympy_to_source(expression, symbols, function_name):
    umbilic_function = sympy.lambdify(
        symbols,
        expression,
        modules=["numpy", "math"],
        # common subexpressions
        cse=True
    )

    return get_function_source(umbilic_function, function_name)


def main() -> int:

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    u_num_control_points = 4
    v_num_control_points = 4

    # Symbolic cross product example:
    #   https://github.com/sympy/sympy/blob/26f7bdbe3f860e7b4492e102edec2d6b429b5aaf/sympy/vector/vector.py#L471

    # Express points under a coordinate system:
    #   https://docs.sympy.org/latest/modules/vector/basics.html#points
    coord_sys = sympy.vector.CoordSys3D('C')

    points_symbol = sympy.Symbol('P')

    # Creates 3 symbols, x, y, and z for each point
    point_component_scalar_symbols = [
        [
            sympy.symbols(f'{points_symbol}[{i}][{j}].x:z')
            for i in range(u_num_control_points)
        ] for j in range(v_num_control_points)
    ]

    # print(point_component_scalar_symbols)

    # Creates points in regards to coordinate system unit vectors.
    #   Allows for doing dot and cross product operations on the vectors
    #       and get a symbolic result.
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

    func_bernstein_polynomial = sympy.Function('B', real=True)

    # Surface equation: Bézier surface
    #   https://en.wikipedia.org/wiki/Bézier_surface
    surface = None
    for j in range(degree_v + 1):
        for i in range(degree_u + 1):
            addition_expr = func_bernstein_polynomial(degree_u, i, u_symbol) \
                * func_bernstein_polynomial(degree_v, j, v_symbol) \
                * points[j][i]

            if surface is None:
                surface = addition_expr
            else:
                surface += addition_expr

    # First fundemental magnitudes
    E_symbol, F_symbol, G_symbol = sympy.symbols('E F G')
    # Second fundemental magnitudes
    L_symbol, M_symbol, N_symbol = sympy.symbols('L M N')

    # Preliminary for First fundemental magnitudes
    u_derivative = sympy.diff(surface, u_symbol)
    v_derivative = sympy.diff(surface, v_symbol)

    # First fundemental magnitudes
    E = u_derivative.dot(u_derivative)
    F = u_derivative.dot(v_derivative)
    G = v_derivative.dot(v_derivative)

    # Preliminary for Second fundemental magnitudes
    normal_vector = u_derivative.cross(v_derivative)
    unit_normal_vector = normal_vector.normalize()

    u_u_derivative = sympy.diff(surface, u_symbol, u_symbol)
    u_v_derivative = sympy.diff(surface, u_symbol, v_symbol)
    v_v_derivative = sympy.diff(surface, v_symbol, v_symbol)

    # Second fundemental magnitudes
    L = unit_normal_vector.dot(u_u_derivative)
    M = unit_normal_vector.dot(u_v_derivative)
    N = unit_normal_vector.dot(v_v_derivative)

    umbilic = (
        (E_symbol * N_symbol)
        + (G_symbol * L_symbol)
        - (2 * F_symbol * M_symbol)
    ) ** 2 - (
        4 * ((L_symbol * N_symbol) - M_symbol ** 2)
        * ((E_symbol * G_symbol) - F_symbol ** 2)
    )

    # print(sympy.pretty(sympy.simplify(umbilic)))
    #
    # print(sympy.pretty(umbilic))

    umbilic = umbilic.subs({
        E_symbol: E,
        F_symbol: F,
        G_symbol: G,
        L_symbol: L,
        M_symbol: M,
        N_symbol: N
    })

    umbilic = umbilic.replace(func_bernstein_polynomial, shared_math.bernstein_polynomial)
    umbilic = umbilic.doit()

    u_derivative_umbilic = sympy.diff(umbilic, u_symbol)
    v_derivative_umbilic = sympy.diff(umbilic, v_symbol)

    print("Lamdify umbilic function")

    output_source = "from math import sqrt"

    output_source += "\n\n" + convert_sympy_to_source(
        umbilic, [points_symbol, u_symbol, v_symbol], "umbilic"
    )

    print("Lamdify umbilic u derivative")

    output_source += '\n\n' + convert_sympy_to_source(
        u_derivative_umbilic, [points_symbol, u_symbol, v_symbol], "umbilic_u_derivative"
    )

    print("Lamdify umbilic v derivative")

    output_source += '\n\n' + convert_sympy_to_source(
        v_derivative_umbilic, [points_symbol, u_symbol, v_symbol], "umbilic_v_derivative"
    )

    with open(OUTPUT_FOLDER / "umbilics.py", 'w') as output_file:
        output_file.write(output_source)

    return 0


if __name__ == '__main__':
    sys.exit(main())
