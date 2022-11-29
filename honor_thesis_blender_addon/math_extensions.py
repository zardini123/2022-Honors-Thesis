import math
import typing
import itertools

# Blender
import mathutils

from . import imported_math_from_sympy
from . import utilities

RAD_TO_DEGREE = 360 / (2 * math.pi)


def lerp(a, b, t):
    """From: https://en.wikipedia.org/wiki/Linear_interpolation"""
    return a + t * (b - a)


def bernstein_polynomial(degree, index, parameter):
    # Binomial args: math.comb(n, k)
    return math.comb(degree, index) \
        * (parameter ** index) \
        * ((1 - parameter) ** (degree - index))


def bezier_surface_at_parameters(control_points, u, v):
    """Equation sourced from: https://en.wikipedia.org/wiki/Bézier_surface"""

    u_num_control_points = len(control_points[0])
    v_num_control_points = len(control_points)

    u_degree = u_num_control_points - 1
    v_degree = v_num_control_points - 1

    output_point = None
    weight_sum = 0
    for i in range(u_degree + 1):
        for j in range(v_degree + 1):
            b_u = bernstein_polynomial(u_degree, i, u)
            b_v = bernstein_polynomial(v_degree, j, v)
            weight = b_u * b_v

            weight_sum += weight

            addition_expr = weight * control_points[j][i]

            if output_point is None:
                output_point = addition_expr
            else:
                output_point += addition_expr

    # Partition of Unity property: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bezier-properties.html
    assert abs(weight_sum - 1) <= 0.00000001

    return output_point


def principle_direction_ret_condition(principle_curvature, patch, u_val, v_val):
    use_direction_1_condition = imported_math_from_sympy.principle_curvature_switch_condition(
        patch, principle_curvature, u_val, v_val
    )

    if use_direction_1_condition:
        principle_direction_u = imported_math_from_sympy.principle_direction_u_1(
            patch, principle_curvature, u_val, v_val
        )

        principle_direction_v = imported_math_from_sympy.principle_direction_v_1(
            patch, principle_curvature, u_val, v_val
        )
    else:
        principle_direction_u = imported_math_from_sympy.principle_direction_u_2(
            patch, principle_curvature, u_val, v_val
        )

        principle_direction_v = imported_math_from_sympy.principle_direction_v_2(
            patch, principle_curvature, u_val, v_val
        )

    principle_vector = mathutils.Vector(
        (principle_direction_u, principle_direction_v))

    return use_direction_1_condition, principle_vector


def principle_direction(principle_curvature, patch, u_val, v_val):
    _, principle_vector = principle_direction_ret_condition(
        principle_curvature, patch, u_val, v_val
    )

    return principle_vector


def principle_direction_arc_length_scaled(principle_curvature, patch, u_val, v_val):
    use_direction_1_condition, principle_vector = principle_direction_ret_condition(
        principle_curvature, patch, u_val, v_val
    )

    if use_direction_1_condition:
        scalar = imported_math_from_sympy.principle_direction_1_arc_length_scalar(
            patch, principle_curvature, u_val, v_val
        )
    else:
        scalar = imported_math_from_sympy.principle_direction_2_arc_length_scalar(
            patch, principle_curvature, u_val, v_val
        )

    return principle_vector * scalar


def min_principle_direction(patch, u_val, v_val, arc_length_scaled=False):
    min_principal_curvature = imported_math_from_sympy.min_principal_curvature(
        patch, u_val, v_val
    )

    if arc_length_scaled:
        min_principal_vector = principle_direction_arc_length_scaled(
            min_principal_curvature, patch, u_val, v_val
        )
    else:
        min_principal_vector = principle_direction(
            min_principal_curvature, patch, u_val, v_val
        )

    return min_principal_vector


def max_principle_direction(patch, u_val, v_val, arc_length_scaled=False):
    max_principal_curvature = imported_math_from_sympy.max_principal_curvature(
        patch, u_val, v_val
    )

    if arc_length_scaled:
        max_principal_vector = principle_direction_arc_length_scaled(
            max_principal_curvature, patch, u_val, v_val
        )
    else:
        max_principal_vector = principle_direction(
            max_principal_curvature, patch, u_val, v_val
        )

    return max_principal_vector


def min_max_principle_directions(patch, u_val, v_val, arc_length_scaled=False):
    min_principal_vector = min_principle_direction(patch, u_val, v_val, arc_length_scaled)
    max_principal_vector = max_principle_direction(patch, u_val, v_val, arc_length_scaled)

    return (min_principal_vector, max_principal_vector)


def is_all_vals_in_bounds(
    values: list[float], lower_inclusive: float, upper_inclusive: float
):
    all_vals_in_bounds = True
    for val in values:
        # print(val)
        if val < lower_inclusive or val > upper_inclusive:
            all_vals_in_bounds = False
            break

    return all_vals_in_bounds


def template_arc_length_continue_func(
    is_start: bool,
    current_parameters: mathutils.Vector,
    num_steps: int,
    current_step_count: int
) -> bool:
    if is_start:
        return True

    if num_steps >= current_step_count:
        return False

    return True


def line_of_curvature_integration_step(
    bezier_patch_control_points,
    current_parameters: mathutils.Vector,
    use_min_principle_direction: bool,
    last_principle_vector=None,
    reverse=False,
):
    if use_min_principle_direction:
        principal_vector = min_principle_direction(
            bezier_patch_control_points,
            current_parameters.x,
            current_parameters.y,
            arc_length_scaled=True
        )
    else:
        principal_vector = max_principle_direction(
            bezier_patch_control_points,
            current_parameters.x,
            current_parameters.y,
            arc_length_scaled=True
        )

    if reverse:
        principal_vector *= -1

    # Flip direction vector when going over ridge to allow integration
    #   to go in one overall direction (instead of oscillating back and forth)
    reverse_due_to_ridge = False
    if last_principle_vector is not None:
        angle = principal_vector.angle(last_principle_vector) * RAD_TO_DEGREE

        if angle > 170:
            print(angle)
            print(principal_vector)
            print(last_principle_vector)

            principal_vector *= -1
            reverse_due_to_ridge = True

    return (principal_vector, reverse_due_to_ridge)


def line_of_curvature_integration_fixed_step_01_bound(
    bezier_patch_control_points,
    current_parameters: mathutils.Vector,
    use_min_principle_direction: bool,
    step_size: float,
    last_principle_vector=None,
    reverse=False
):
    principle_vector, reverse_due_to_ridge = line_of_curvature_integration_step(
        bezier_patch_control_points,
        current_parameters,
        use_min_principle_direction,
        last_principle_vector=last_principle_vector,
        reverse=reverse
    )

    next_parameters = current_parameters + (principle_vector * step_size)

    bottom_left = mathutils.Vector((0, 0))
    bottom_right = mathutils.Vector((1, 0))

    top_left = mathutils.Vector((0, 1))
    top_right = mathutils.Vector((1, 1))

    val_in_bounds = True
    for pair in itertools.product([bottom_left, top_right], [top_left, bottom_right], repeat=2):
        intersection = mathutils.geometry.intersect_line_line_2d(
            pair[0],
            pair[1],
            current_parameters,
            next_parameters
        )

        if intersection is not None:
            val_in_bounds = False
            next_parameters = intersection

    # val_in_bounds = is_all_vals_in_bounds(
    #     [
    #         next_parameters.x,
    #         next_parameters.y,
    #     ],
    #     0, 1
    # )

    return (val_in_bounds, principle_vector, next_parameters, reverse_due_to_ridge)


def integrate_lines_of_curvature_image(
    image_buffer,
    image_size,
    color,
    bezier_patch_control_points,
    start_parameters: tuple[float, float],
    arc_length_distance: float,
    step_size: float,
    use_min_principle_direction=False,
    reverse=False
):

    current_principle_vector = None
    current_parameters = mathutils.Vector(start_parameters)

    utilities.rasterize_point(
        image_buffer,
        current_parameters,
        image_size,
        color
    )

    for step in range(int(math.ceil(arc_length_distance / step_size))):
        val_in_bounds, next_principle_vector, next_parameters, _ = \
            line_of_curvature_integration_fixed_step_01_bound(
                bezier_patch_control_points,
                current_parameters,
                use_min_principle_direction,
                step_size,
                last_principle_vector=current_principle_vector,
                reverse=reverse
            )

        utilities.rasterize_point(
            image_buffer,
            next_parameters,
            image_size,
            color
        )

        if not val_in_bounds:
            break

        current_parameters = next_parameters
        current_principle_vector = next_principle_vector


def integrate_line_of_curvature_to_mesh_edges(
    output_bmesh,
    bezier_patch_control_points,
    start_parameters: tuple[float, float],
    arc_length_distance: float,
    step_size: float,
    use_min_principle_direction: bool,
    reverse=False
):
    current_vert = output_bmesh.verts.new(bezier_surface_at_parameters(
        bezier_patch_control_points, start_parameters.x, start_parameters.y
    ))

    final_verts = []

    current_principle_vector = None
    current_parameters = mathutils.Vector(start_parameters)

    for step in range(int(math.ceil(arc_length_distance / step_size))):
        val_in_bounds, next_principle_vector, next_parameters, _ = \
            line_of_curvature_integration_fixed_step_01_bound(
                bezier_patch_control_points,
                current_parameters,
                use_min_principle_direction,
                step_size,
                last_principle_vector=current_principle_vector,
                reverse=reverse
            )

        if not val_in_bounds:
            break

        next_vert = output_bmesh.verts.new(bezier_surface_at_parameters(
            bezier_patch_control_points, next_parameters.x, next_parameters.y
        ))

        vert = output_bmesh.edges.new((next_vert, current_vert))

        final_verts.append(vert)

        current_vert = next_vert
        current_parameters = next_parameters
        current_principle_vector = next_principle_vector

    return final_verts


def umbilic_gradient_descent(
    start_parameters: mathutils.Vector,
    control_points,
    iterations=1000,
    learning_rate=1e-5,
    stopping_threshold=1e-6
):
    previous_parameters = start_parameters
    previous_value = imported_math_from_sympy.umbilic(
        control_points,
        start_parameters.x,
        start_parameters.y
    )

    for iteration in range(iterations):
        u_derivative = imported_math_from_sympy.umbilic_u_derivative(
            control_points,
            previous_parameters.x,
            previous_parameters.y
        )
        v_derivative = imported_math_from_sympy.umbilic_v_derivative(
            control_points,
            previous_parameters.x,
            previous_parameters.y
        )

        gradient = mathutils.Vector((
            u_derivative,
            v_derivative,
        ))

        # print(gradient.magnitude)
        # print(gradient.normalized())

        scaled_gradient = gradient

        # print(previous_parameters)

        current_parameters = previous_parameters - (learning_rate * gradient)

        if current_parameters.x < 0 or current_parameters.x > 1 or current_parameters.y < 0 or current_parameters.y > 1:
            return False, previous_value, previous_parameters

        # print(current_parameters)

        current_value = imported_math_from_sympy.umbilic(
            control_points,
            current_parameters.x,
            current_parameters.y
        )

        if abs(current_value - previous_value) <= stopping_threshold:
            return True, current_value, current_parameters

        previous_parameters = current_parameters
        previous_value = current_value

    return False, current_value, current_parameters
