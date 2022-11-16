"""2022 Honors Thesis Blender Addon."""

import math
import statistics
import typing
import datetime

import numpy

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add

import mathutils

# "Multi files to addon":
#   https://blender.stackexchange.com/a/202672/76575
# "Blender won't import my custom module, ImportError: No module named mymodule"
#   https://blender.stackexchange.com/a/42192/76575
import importlib
if 'utilities' in globals():
    importlib.reload(utilities)
if 'imported_math_from_sympy' in globals():
    importlib.reload(imported_math_from_sympy)

from . import utilities
from . import imported_math_from_sympy

bl_info = {
    "name": "2022 Honors Thesis",
    "description": "Bézier surfaces with meshing",
    "author": "Taras Palczynski III",
    "version": (0, 0, 1),
    "blender": (3, 3, 0),
    "location": "View3D > Add > Mesh",
    "warning": "",  # used for warning icon and text in addons panel
    "doc_url": "https://github.com/zardini123/2022-Honors-Thesis",
    "tracker_url": "https://github.com/zardini123/2022-Honors-Thesis/issues",
    "support": "COMMUNITY",
    "category": "Mesh",
}

RAD_TO_DEGREE = 360 / (2 * math.pi)


def add_object(self, context):
    scale_x = self.scale.x
    scale_y = self.scale.y

    def generator_function(u, v):
        x_pos = (u * 2) - 1
        y_pos = (v * 2) - 1

        return mathutils.Vector((x_pos * scale_x, y_pos * scale_y, 0))

    verticies, faces = utilities.grid_mesh(generator_function, 4, 4)

    control_grid_mesh = bpy.data.meshes.new(name="Bézier Control Mesh")

    # output_mesh will not be populated with anything (blank mesh)
    output_mesh = bpy.data.meshes.new(name="Bézier Output Mesh")

    edges = []

    # from_pydata:
    #   https://docs.blender.org/api/current/bpy.types.Mesh.html#bpy.types.Mesh.from_pydata
    control_grid_mesh.from_pydata(verticies, edges, faces)

    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    # object_data_add:
    #   https://docs.blender.org/api/current/bpy_extras.object_utils.html#bpy_extras.object_utils.object_data_add
    created_control_grid = object_data_add(
        context, control_grid_mesh, operator=self, name="Bézier Surface")
    created_output_object = object_data_add(
        context, output_mesh, operator=self, name="Bézier Output Object")

    # Set boundary edges as sharp
    control_grid_bmesh = bmesh.new()   # create an empty BMesh
    control_grid_bmesh.from_mesh(control_grid_mesh)

    # Get boundary edges: https://blender.stackexchange.com/a/242988/76575
    control_grid_bmesh.edges.ensure_lookup_table()
    for edge in control_grid_bmesh.edges:
        if edge.is_boundary:
            edge.smooth = False

    control_grid_bmesh.to_mesh(control_grid_mesh)

    # Setup output object metadata
    created_output_object.parent = created_control_grid

    # Used Python tooltip feature of Blender UI to determine these API calls
    created_control_grid.display.show_shadows = False
    created_control_grid.show_wire = True
    created_control_grid.display_type = 'WIRE'
    created_control_grid.hide_render = True

    created_output_object.hide_select = True


class OBJECT_OT_add_bezier_surface(Operator, AddObjectHelper):
    """Create a new Mesh Object"""
    bl_idname = "surface.bezier_surface"
    bl_label = "Add Bézier Surface"
    bl_options = {'REGISTER', 'UNDO'}

    scale: FloatVectorProperty(
        name="Scale",
        default=(1.0, 1.0, 1.0),
        subtype='TRANSLATION',
        description="Scaling",
    )

    def execute(self, context):
        add_object(self, context)
        return {'FINISHED'}


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


def add_value_to_array(array, value):
    return [element + value for element in array]


def ensure_uv_layers(
    uv_layers: bmesh.types.BMLayerCollection,
    num_uv_layers: int
) -> list[bmesh.types.BMLayerItem]:
    output_uv_layers = [None] * num_uv_layers
    patch_prefix = "Patch_"

    # Acquire existing layers
    # @FIXME: BMesh created from scratch each edit.
    #   Preserve previous UV properties when creating new BMesh

    # for uv_layer_name, uv_layer in uv_layers.items():
    #     print(uv_layer_name)

    # Create new layers for ones not existing
    for i, output_uv_layer in enumerate(output_uv_layers):
        if output_uv_layer is not None:
            continue

        output_uv_layers[i] = uv_layers.new(f"{patch_prefix}{i}")

    return output_uv_layers


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


def lerp(a, b, t):
    """From: https://en.wikipedia.org/wiki/Linear_interpolation"""
    return a + t * (b - a)


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


def create_and_replace_output_mesh(control_mesh: bpy.types.Mesh, output_object: bpy.types.Object):
    control_bmesh = bmesh.from_edit_mesh(control_mesh)

    # print("start\n")
    patches_sharp_corner_faces = \
        utilities.get_3x3_patches_from_sharp_border(control_bmesh)

    patches = []
    for sharp_corner_faces in patches_sharp_corner_faces:
        control_loops = \
            utilities.get_3x3_patch_loops_in_topological_order(sharp_corner_faces)

        control_verticies = [
            [loop.vert.co for loop in x_array] for x_array in control_loops
        ]

        patches.append(control_verticies)

    start_time = datetime.datetime.now()

    if True:
        # Start new mesh from scratch
        output_bmesh = bmesh.new()

        u_num_verts = v_num_verts = 20

        # @NOTE: Cannot have mulitple UV Maps as only "one" UV map is able to be
        #   inputted to materials without Nodes.  Therefore use material index
        #   to differentiate textures in patches.
        # num_patches = len(patches)
        # uv_layers = ensure_uv_layers(output_bmesh.loops.layers.uv, num_patches)
        # print(uv_layers)

        uv_layer = output_bmesh.loops.layers.uv.verify()

        for patch_index, patch_control_points in enumerate(patches):
            def generator_function(u, v):
                return bezier_surface_at_parameters(patch_control_points, u, v)

            # @TODO: Each grid mesh is not connected, new verticies are made for
            #   each bordering patch
            added_verticies_references, added_faces_references = \
                utilities.add_grid_mesh_to_bmesh(
                    output_bmesh, generator_function, u_num_verts, v_num_verts
                )

            # Create UV Map for patch going from 0 to 1 on both U and V for all
            #   patches
            for vertex_index, vertex_reference in enumerate(added_verticies_references):
                v_index = vertex_index % u_num_verts
                u_index = vertex_index // u_num_verts

                u_pos = u_index / (u_num_verts - 1)
                v_pos = v_index / (v_num_verts - 1)

                for loop in vertex_reference.link_loops:
                    # @TODO: Why does array access on a loop for uv work??
                    loop[uv_layer].uv = mathutils.Vector((u_pos, v_pos))

            # Set material index for patch for seperating each patch's material
            for face in added_faces_references:
                face.material_index = patch_index

        # Ensure adequate number of material slots
        num_patches = len(patches)
        starting_num_materials = len(output_object.data.materials)
        for i in range(starting_num_materials, num_patches):
            output_object.data.materials.append(None)

        patch_prefix = "Patch_"

        u_image_size = 1024
        v_image_size = 1024

        # Ensure adequate number of image data blocks for each patch
        images = [None] * num_patches
        for image_name, image in bpy.data.images.items():
            if patch_prefix in image_name:
                if image.size[0] != u_image_size and image.size[1] != v_image_size:
                    bpy.data.images.remove(image)
                else:
                    index = int(image_name.replace(patch_prefix, ""))
                    images[index] = image

        for image_index, image in enumerate(images):
            if image is None:
                images[image_index] = bpy.data.images.new(
                    f"{patch_prefix}{image_index}", u_image_size, v_image_size
                )

        # Create new material for all material slots non-ocupied
        #   that has the image for the patch UV-mapped.
        for patch_index, material in enumerate(output_object.data.materials):
            # Ensure material slots have a material present
            if material is None:
                new_material = bpy.data.materials.new(
                    f"{patch_prefix}{patch_index}"
                )

                # Create material with texture from Python:
                #   https://blender.stackexchange.com/a/240372/76575
                new_material.use_nodes = True

                node_tree = new_material.node_tree
                nodes = new_material.node_tree.nodes
                nodes.clear()

                node_uv_map = nodes.new(type='ShaderNodeUVMap')
                node_uv_map.location = (-800, 0)

                node_texture = nodes.new('ShaderNodeTexImage')
                node_texture.image = images[patch_index]
                node_texture.location = (-400, 0)

                node_diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
                node_diffuse.location = (0, 0)

                node_output = nodes.new(type='ShaderNodeOutputMaterial')
                node_output.location = (400, 0)

                links = new_material.node_tree.links
                link = links.new(node_uv_map.outputs["UV"], node_texture.inputs["Vector"])
                link = links.new(node_texture.outputs["Color"], node_diffuse.inputs["Color"])
                link = links.new(node_diffuse.outputs["BSDF"], node_output.inputs["Surface"])

                output_object.data.materials[patch_index] = new_material
                material = new_material

            # Ignore materials who dont use nodes
            # if not material.use_nodes:
            #     break

            # found_image_node = False
            # for node in material.node_tree.nodes:
            #     if node.name == "Image Texture":
            #         found_image_node = True
            #         break

            # if not found_image_node:
            #     texImage = material.node_tree.nodes.new('ShaderNodeTexImage')
            #     texImage.image

        # start_time = datetime.datetime.now()

        if True:
            image_buffer = numpy.zeros((v_image_size, u_image_size, 4))

            u_num_sample_points = 15
            v_num_sample_points = 15

            u_scan_size = 10
            v_scan_size = 10

            # Modify textures to display surface values
            for patch_index, image in enumerate(images):
                num_data_points = len(image.pixels)
                u_size, v_size = image.size
                num_channels = image.channels

                if True:
                    start_depth = 0

                    stack = []
                    stack.append(
                        (start_depth, False, False, (0, 1), (0, 1))
                    )

                    is_parallel_angle_threshold = 20
                    too_small_size_of_bound_threshold = 1e-14
                    max_depth = 8  # 10

                    umbilic_threshold = 1e-10
                    is_ridge_angle_threshold = 170

                    same_direction_magnitude_threshold = 0.999
                    ridge_magnitude_threshold = 0.6

                    num_samples_per_axis = 2

                    area_percentages = {}

                    depth = []

                    while len(stack) != 0:
                        stack_tuple = stack.pop()
                        current_depth, has_umbilic, has_ridge, u_bound, v_bound = stack_tuple

                        if abs(u_bound[0] - u_bound[1]) < too_small_size_of_bound_threshold:
                            continue

                        if abs(v_bound[0] - v_bound[1]) < too_small_size_of_bound_threshold:
                            continue

                        if current_depth > 1000:
                            print("ERROR: MAX STACK DEPTH HIT")
                            stack = []
                            break

                        # print(stack_tuple)

                        # Determine if principle direction orientations are generally
                        #   in the same orientation

                        min_p_running_average = None
                        max_p_running_average = None

                        need_subdivide = False
                        is_umbilic = False
                        is_ridge = False

                        min_p_summed_vectors = mathutils.Vector((0, 0))
                        max_p_summed_vectors = mathutils.Vector((0, 0))
                        count = 0

                        min_principle_directions = set()
                        max_principle_directions = set()

                        min_previous_vector = None
                        max_previous_vector = None

                        for v_index in range(num_samples_per_axis):
                            if need_subdivide:
                                break

                            for u_index in range(num_samples_per_axis):
                                u_pre = u_index / (num_samples_per_axis - 1)
                                v_pre = v_index / (num_samples_per_axis - 1)

                                u = lerp(u_bound[0], u_bound[1], u_pre)
                                v = lerp(v_bound[0], v_bound[1], v_pre)

                                # print(u, v)

                                umbilic_val = imported_math_from_sympy.umbilic(
                                    patches[patch_index], u, v
                                )

                                # print("umbilic val:", umbilic_val)

                                if umbilic_val <= 0.1:
                                    is_umbilic = True

                                # if umbilic_val <= umbilic_threshold:
                                #     print("umbilic:", current_depth, umbilic_val)
                                #     need_subdivide = False
                                #     is_umbilic = True
                                #     break

                                min_principal_vector, max_principal_vector = \
                                    min_max_principle_directions(
                                        patches[patch_index], u, v
                                    )

                                # world_point = bezier_surface_at_parameters(
                                #     patches[patch_index], u, v
                                # )
                                # world_min_principal_vector = bezier_surface_at_parameters(
                                #     patches[patch_index],
                                #     u + min_principal_vector.x,
                                #     v + min_principal_vector.y
                                # )
                                # world_max_principal_vector = bezier_surface_at_parameters(
                                #     patches[patch_index],
                                #     u + max_principal_vector.x,
                                #     v + max_principal_vector.y
                                # )
                                #
                                # min_principal_vector = world_min_principal_vector - world_point
                                # max_principal_vector = world_max_principal_vector - world_point

                                if True:
                                    min_principle_directions.add(
                                        min_principal_vector.normalized().freeze())
                                    max_principle_directions.add(
                                        max_principal_vector.normalized().freeze())

                                    min_p_summed_vectors += min_principal_vector.normalized()
                                    max_p_summed_vectors += max_principal_vector.normalized()

                                    count += 1

                                # if True:
                                #     if min_previous_vector is None:
                                #         min_previous_vector = min_principal_vector
                                #
                                #     if max_previous_vector is None:
                                #         max_previous_vector = max_principal_vector

                                if False:
                                    if count == 0:
                                        min_p_running_average = min_principal_vector
                                        max_p_running_average = max_principal_vector

                                        count += 1
                                        continue

                                    rad_to_degree = 360 / (2 * math.pi)

                                    min_current_angle = min_p_running_average.angle(
                                        min_principal_vector) * rad_to_degree
                                    max_current_angle = max_p_running_average.angle(
                                        max_principal_vector) * rad_to_degree

                                    # print(min_current_angle, max_current_angle)

                                    # Vector not parallel enough
                                    # -90, 90 parellel
                                    # 20 degree threshold
                                    # [-90, -70] and [70, 90]
                                    # If in [0, 70] range, is not parallel enough

                                    if min_current_angle > is_ridge_angle_threshold \
                                            or max_current_angle > is_ridge_angle_threshold:
                                        is_ridge = True

                                    if current_depth < max_depth:
                                        # if min_current_angle > is_parallel_angle_threshold \
                                        #         or max_current_angle > is_parallel_angle_threshold:
                                        if abs(min_current_angle - 90) < (90 - is_parallel_angle_threshold) \
                                                or abs(max_current_angle - 90) < (90 - is_parallel_angle_threshold):

                                            # if abs(min_current_angle - 90) < (90 - is_parallel_angle_threshold):  # \
                                            # if abs(max_current_angle - 90) < (90 - is_parallel_angle_threshold):
                                            # or abs(max_current_angle - 90) < (90 - is_parallel_angle_threshold):
                                            need_subdivide = True
                                            break

                                    # If pointing opposite direction, maybe sampling across a ridge
                                    # if min_current_dot < 0:
                                    #     min_principal_vector *= -1
                                    # if max_current_dot < 0:
                                    #     max_principal_vector *= -1

                        if True:
                            min_p_average_vector = min_p_summed_vectors / count
                            max_p_average_vector = max_p_summed_vectors / count

                            # print("min:", min_p_average_vector.magnitude)
                            # print("max:", max_p_average_vector.magnitude)

                            ridge = False
                            near_umbilic = False

                            min_deviating = False
                            max_deviating = False

                            # print(min_p_average_vector.magnitude)
                            # print(max_p_average_vector.magnitude)

                            # Ridge is opposite direction, therefore results in near 0 magnitude
                            if min_p_average_vector.magnitude < ridge_magnitude_threshold:
                                if max_p_average_vector.magnitude > same_direction_magnitude_threshold:
                                    ridge = True
                                    min_deviating = True
                            else:
                                # If less magnitude (more chaos) in min direction but not enough for being a ridge, near umbilic
                                if min_p_average_vector.magnitude < same_direction_magnitude_threshold:
                                    near_umbilic = True
                                    min_deviating = True

                            if max_p_average_vector.magnitude < ridge_magnitude_threshold:
                                if min_p_average_vector.magnitude > same_direction_magnitude_threshold:
                                    ridge = True
                                    max_deviating = True
                            else:
                                if max_p_average_vector.magnitude < same_direction_magnitude_threshold:
                                    near_umbilic = True
                                    max_deviating = True

                            # if not ridge and not umbilic and max_p_average_vector.magnitude < same_direction_magnitude_threshold:
                            #     need_subdivide = True

                            # if ridge and not umbilic:
                            #     print("ridge:", ridge)
                            #     print("umbilic:", umbilic)

                            if ridge or near_umbilic:
                                if current_depth < max_depth:
                                    need_subdivide = True
                                else:
                                    current_depth += 1

                        # print(need_subdivide)

                        if need_subdivide:
                            next_depth = current_depth + 1
                            half_u = (u_bound[0] + u_bound[1]) / 2
                            half_v = (v_bound[0] + v_bound[1]) / 2

                            new_has_umbilic = True if near_umbilic else has_umbilic
                            new_has_ridge = True if ridge else has_ridge

                            # Bottom left
                            stack.append(
                                (next_depth, new_has_umbilic, new_has_ridge,
                                 (u_bound[0], half_u), (v_bound[0], half_v))
                            )
                            # Bottom right
                            stack.append(
                                (next_depth, new_has_umbilic, new_has_ridge,
                                 (half_u, u_bound[1]), (v_bound[0], half_v))
                            )
                            # Top Left
                            stack.append(
                                (next_depth, new_has_umbilic, new_has_ridge,
                                 (u_bound[0], half_u), (half_v, v_bound[1]))
                            )
                            # Top Right
                            stack.append(
                                (next_depth, new_has_umbilic, new_has_ridge,
                                 (half_u, u_bound[1]), (half_v, v_bound[1]))
                            )
                        else:
                            width = u_bound[1] - u_bound[0]
                            height = v_bound[1] - v_bound[0]
                            area = width * height

                            if current_depth not in area_percentages:
                                area_percentages[current_depth] = 0

                            area_percentages[current_depth] += area
                            depth.append(current_depth)

                            u_pixel_start = int(round(u_size * u_bound[0]))
                            u_pixel_end = int(round(u_size * u_bound[1]))

                            v_pixel_start = int(round(u_size * v_bound[0]))
                            v_pixel_end = int(round(u_size * v_bound[1]))

                            for v_pixel in range(v_pixel_start, v_pixel_end):
                                for u_pixel in range(u_pixel_start, u_pixel_end):
                                    r = (current_depth * 0.3) % 1.0
                                    g = (current_depth * 0.5) % 1.0
                                    b = (current_depth * 0.7) % 1.0

                                    # vector = min_p_average_vector.normalized()
                                    # vector = max_p_average_vector.normalized()
                                    # r = (vector.x + 1) / 2
                                    # g = (vector.y + 1) / 2
                                    # b = 0.0

                                    # if is_umbilic:
                                    #     r = 1.0
                                    #     g = b = 0.0
                                    # else:
                                    #     r = g = b = 0.0
                                    #
                                    # if is_ridge:
                                    #     g = 1.0

                                    # if umbilic or ridge:
                                    #     r = g = b = 0.0

                                    r = g = b = 0.0

                                    # if near_umbilic:
                                    #     r = 1.0

                                    # if ridge:
                                    #     g = 1.0

                                    if min_deviating:
                                        b = 1.0
                                        g = 0.5
                                    if max_deviating:
                                        r = 1.0

                                    image_buffer[u_pixel][v_pixel] = (
                                        r, g, b, 1.0
                                    )

                    print(area_percentages)
                    print(statistics.mean(area_percentages.values()))
                    print(statistics.stdev(area_percentages.values()))

                if False:
                    for v_pixel in range(v_size):
                        for u_pixel in range(u_size):
                            u_val = u_pixel / (u_size - 1)
                            v_val = v_pixel / (v_size - 1)

                            min_principal_curvature = imported_math_from_sympy.min_principal_curvature(
                                patches[patch_index], u_val, v_val
                            )

                            max_principal_curvature = imported_math_from_sympy.max_principal_curvature(
                                patches[patch_index], u_val, v_val
                            )

                            principle_curvature = min_principal_curvature

                            use_direction_1_condition = imported_math_from_sympy.principle_curvature_switch_condition(
                                patches[patch_index], principle_curvature, u_val, v_val
                            )

                            if use_direction_1_condition:
                                principle_direction_u = imported_math_from_sympy.principle_direction_u_1(
                                    patches[patch_index], principle_curvature, u_val, v_val
                                )

                                principle_direction_v = imported_math_from_sympy.principle_direction_v_1(
                                    patches[patch_index], principle_curvature, u_val, v_val
                                )
                            else:
                                principle_direction_u = imported_math_from_sympy.principle_direction_u_2(
                                    patches[patch_index], principle_curvature, u_val, v_val
                                )

                                principle_direction_v = imported_math_from_sympy.principle_direction_v_2(
                                    patches[patch_index], principle_curvature, u_val, v_val
                                )

                            principle_vector = mathutils.Vector(
                                (principle_direction_u, principle_direction_v))
                            principle_vector.normalize()

                            # if not switch_condition:
                            #     r = g = b = 1.0
                            # else:
                            #     r = g = b = 0.0

                            r = ((principle_vector.x + 1) / 2)
                            g = ((principle_vector.y + 1) / 2)
                            b = 0.0

                            image_buffer[u_pixel][v_pixel] = (
                                r, g, b, 1.0
                            )

                if False:
                    for v_pixel in range(v_size):
                        for u_pixel in range(u_size):
                            u_val = u_pixel / (u_size - 1)
                            v_val = v_pixel / (v_size - 1)

                            min_principal_curvature = imported_math_from_sympy.min_principal_curvature(
                                patches[patch_index], u_val, v_val
                            )

                            max_principal_curvature = imported_math_from_sympy.max_principal_curvature(
                                patches[patch_index], u_val, v_val
                            )

                            # print(min_principal_curvature)
                            # print(max_principal_curvature)

                            if min_principal_curvature != 0:
                                # Pixel is umbilic when min == max (ratio is 1 (not -1))
                                ratio = max_principal_curvature / min_principal_curvature

                                # @TODO: Very interesting:
                                # ratio = abs(max_principal_curvature) / abs(min_principal_curvature)

                                # When ratio = 1, both values the same

                                # min clamps range to [0, 1]
                                distance_ratio_to_1 = min(abs(ratio - 1), 1)

                                # White pixel is umbilic (when distance = 1)
                                when_ratio_is_1_this_is_1 = 1 - distance_ratio_to_1

                                if when_ratio_is_1_this_is_1 >= 0.9:
                                    image_buffer[u_pixel][v_pixel] = (
                                        when_ratio_is_1_this_is_1, 0.0, 0.0, 1.0
                                    )
                                else:
                                    image_buffer[u_pixel][v_pixel] = (
                                        when_ratio_is_1_this_is_1,
                                        when_ratio_is_1_this_is_1,
                                        when_ratio_is_1_this_is_1,
                                        1.0
                                    )

                if False:
                    for v_pixel in range(v_size):
                        for u_pixel in range(u_size):
                            u_val = u_pixel / (u_size - 1)
                            v_val = v_pixel / (v_size - 1)

                            umbilic_val = imported_math_from_sympy.umbilic(
                                patches[patch_index], u_val, v_val
                            )

                            r = g = b = 0.0

                            # umbilic_val = abs(umbilic_val)

                            if umbilic_val <= 1.0:
                                r = 1.0
                                g = 1.0
                                b = 1.0
                            elif umbilic_val > 1.0 and umbilic_val <= 200.0:
                                r = umbilic_val / 200.0
                            elif umbilic_val > 200.0 and umbilic_val <= 400.0:
                                g = 1.0
                            elif umbilic_val > 400.0:
                                b = 1.0

                            image_buffer[u_pixel][v_pixel] = (
                                r, g, b, 1.0
                            )

                            # if umbilic_val <= 1.0:
                            #
                            #     u_pixel_pos = int((u_size - 1) * u_val)
                            #     v_pixel_pos = int((v_size - 1) * u_val)
                            #
                            #     image_buffer[v_pixel_pos][u_pixel_pos] = (
                            #         1.0, 0.0, 0.0, 1.0
                            #     )

                if False:
                    for v_sample_point in range(v_num_sample_points):
                        for u_sample_point in range(u_num_sample_points):
                            u_pos = u_sample_point / (u_num_sample_points - 1)
                            v_pos = v_sample_point / (v_num_sample_points - 1)

                            found, value, parameters = umbilic_gradient_descent(
                                mathutils.Vector((u_pos, v_pos)),
                                patches[patch_index]
                            )

                            u_pixel_pos = int((u_size - 1) * parameters.x)
                            v_pixel_pos = int((v_size - 1) * parameters.y)

                            r = g = b = 0.0

                            if found:
                                if value <= 1e-6:
                                    r = g = b = 1.0
                                else:
                                    r = 1.0
                            else:
                                if value <= 1e-6:
                                    b = 1.0
                                else:
                                    g = 1.0

                            print(value)

                            image_buffer[u_pixel_pos][v_pixel_pos] = (
                                r, g, b, 1.0
                            )

                if False:
                    for v_pixel in range(v_scan_size):
                        for u_pixel in range(u_scan_size):
                            u_val = u_pixel / (u_scan_size - 1)
                            v_val = v_pixel / (v_scan_size - 1)

                            # umbilic_val = imported_math_from_sympy.umbilic(
                            #     patches[patch_index], u_val, v_val
                            # )
                            #
                            # if umbilic_val <= 100:

                            found, value, parameters = umbilic_gradient_descent(
                                mathutils.Vector((u_val, v_val)),
                                patches[patch_index],
                                iterations=1000,
                                learning_rate=1e-5,
                                stopping_threshold=1e-6
                            )

                            u_pixel_pos = int((u_size - 1) * parameters.x)
                            v_pixel_pos = int((v_size - 1) * parameters.y)

                            r = g = b = 0.0

                            compare_val = 1e-4

                            true_val = imported_math_from_sympy.umbilic(
                                patches[patch_index],
                                parameters.x,
                                parameters.y
                            )

                            print(value, true_val)

                            if value <= compare_val:
                                image_buffer[u_pixel_pos][v_pixel_pos] = (
                                    0.0, 1.0, 0.0, 1.0
                                )

                            # if found:
                            #     if value <= compare_val:
                            #         r = g = b = 1.0
                            #     else:
                            #         r = 1.0
                            # else:
                            #     if value <= compare_val:
                            #         b = 1.0
                            #     else:
                            #         g = 1.0
                            #
                            # print(found, value)
                            #
                            # image_buffer[u_pixel_pos][v_pixel_pos] = (
                            #     r, g, b, 1.0
                            # )

                raveled_image = numpy.ravel(image_buffer)
                image.pixels = raveled_image
                image.update()

                # output_pixels = [0.0] * u_size * v_size * num_channels
                # for v_pixel in range(v_size):
                #     for u_pixel in range(u_size):
                #         pixel_start = (v_pixel * u_size * num_channels) + u_pixel * num_channels
                #         output_pixels[pixel_start: pixel_start + num_channels] = (1.0, 1.0, 1.0, 1.0)

                # for index in range(num_data_points):
                #     image.pixels[index] = 0.0

                # print(len(image.pixels))
                # print(image.channels)

        # end_time = datetime.datetime.now()
        # time_lapsed = (end_time - start_time).total_seconds()
        # print(time_lapsed)

        # Overwrite mesh
        output_mesh = output_object.data
        output_bmesh.to_mesh(output_mesh)

    if False:
        # Start new mesh from scratch
        output_bmesh = bmesh.new()

        output_object.vertex_groups.clear()
        # https://blender.stackexchange.com/a/117586
        vertex_group = output_object.vertex_groups.new(name="Max direction").index
        weight_layer = output_bmesh.verts.layers.deform.new("Max direction")

        for patch in patches:
            u_samples = v_samples = 200  # 50

            for v_sample in range(v_samples):
                for u_sample in range(u_samples):
                    u_val = u_sample / (u_samples - 1)
                    v_val = v_sample / (v_samples - 1)

                    world_point = bezier_surface_at_parameters(patch, u_val, v_val)

                    min_principal_vector, max_principal_vector = \
                        min_max_principle_directions(
                            patch, u_val, v_val
                        )

                    test_orthogonality = False

                    if test_orthogonality:
                        orthogonal_verification_value = imported_math_from_sympy.orthogonal_verification_value(
                            patch, u_val, v_val,
                            min_principal_vector.x, min_principal_vector.y,
                            max_principal_vector.x, max_principal_vector.y,
                        )

                        is_orthogonal = abs(orthogonal_verification_value) <= 0.0001
                        if not is_orthogonal:
                            print("NOT ORTHOGONAL")

                    vector_scale = 0.0025  # * 3
                    min_principal_vector = min_principal_vector.normalized() * vector_scale
                    max_principal_vector = max_principal_vector.normalized() * vector_scale

                    parameters_vector = mathutils.Vector((u_val, v_val))
                    min_vector_end_parameters = parameters_vector + min_principal_vector
                    max_vector_end_parameters = parameters_vector + max_principal_vector

                    min_vector_end = bezier_surface_at_parameters(
                        patch, min_vector_end_parameters.x, min_vector_end_parameters.y
                    )

                    max_vector_end = bezier_surface_at_parameters(
                        patch, max_vector_end_parameters.x, max_vector_end_parameters.y
                    )

                    enable_min = False
                    enable_max = True

                    if enable_min:
                        start_vert_1 = output_bmesh.verts.new(world_point)
                        min_end_vert = output_bmesh.verts.new(min_vector_end)

                        start_vert_1[weight_layer][vertex_group] = 0.0
                        min_end_vert[weight_layer][vertex_group] = 0.0

                        output_bmesh.edges.new((start_vert_1, min_end_vert))

                    if enable_max:
                        start_vert_2 = output_bmesh.verts.new(world_point)
                        max_end_vert = output_bmesh.verts.new(max_vector_end)

                        start_vert_2[weight_layer][vertex_group] = 1.0
                        max_end_vert[weight_layer][vertex_group] = 1.0

                        output_bmesh.edges.new((start_vert_2, max_end_vert))

        output_mesh = output_object.data
        output_bmesh.to_mesh(output_mesh)

    if False:
        # Start new mesh from scratch
        output_bmesh = bmesh.new()

        color_layer = output_bmesh.verts.layers.color.new("Min/Max Color")

        for patch in patches:
            u_samples = v_samples = 10

            arc_length_distance = 1
            step_size = 0.001

            for v_sample in range(v_samples):
                for u_sample in range(u_samples):
                    u_val = u_sample / (u_samples - 1)
                    v_val = v_sample / (v_samples - 1)

                    min_last_parameters = mathutils.Vector((u_val, v_val))
                    max_last_parameters = mathutils.Vector((u_val, v_val))

                    min_last_vert = output_bmesh.verts.new(bezier_surface_at_parameters(
                        patch, min_last_parameters.x, min_last_parameters.y
                    ))
                    max_last_vert = output_bmesh.verts.new(bezier_surface_at_parameters(
                        patch, max_last_parameters.x, max_last_parameters.y
                    ))

                    for step in range(int(math.ceil(arc_length_distance / step_size))):
                        # print(step)
                        min_principal_vector = min_principle_direction(
                            patch, min_last_parameters.x, min_last_parameters.y, arc_length_scaled=True
                        )
                        max_principal_vector = max_principle_direction(
                            patch, max_last_parameters.x, max_last_parameters.y, arc_length_scaled=True
                        )

                        min_current_parameters = min_last_parameters + \
                            (min_principal_vector * step_size)
                        max_current_parameters = max_last_parameters + \
                            (max_principal_vector * step_size)

                        vals_to_check = [
                            min_current_parameters.x,
                            min_current_parameters.y,
                            max_current_parameters.x,
                            max_current_parameters.y
                        ]
                        val_bad = False
                        for val in vals_to_check:
                            # print(val)
                            if val < 0 or val > 1:
                                val_bad = True
                                break

                        if val_bad:
                            break

                        min_current_vert = output_bmesh.verts.new(bezier_surface_at_parameters(
                            patch, min_current_parameters.x, min_current_parameters.y
                        ))
                        max_current_vert = output_bmesh.verts.new(bezier_surface_at_parameters(
                            patch, max_current_parameters.x, max_current_parameters.y
                        ))

                        output_bmesh.edges.new((min_current_vert, min_last_vert))
                        output_bmesh.edges.new((max_current_vert, max_last_vert))

                        min_current_vert[color_layer] = (0.188, 0.439, 0.945, 1.0)
                        max_current_vert[color_layer] = (0.878, 0.298, 0.298, 1.0)

                        min_last_vert = min_current_vert
                        max_last_vert = max_current_vert

                        min_last_parameters = min_current_parameters
                        max_last_parameters = max_current_parameters

        output_mesh = output_object.data
        output_bmesh.to_mesh(output_mesh)

    end_time = datetime.datetime.now()
    time_lapsed = (end_time - start_time).total_seconds()
    print(time_lapsed)


def cb_scene_update(scene):
    edit_obj = bpy.context.edit_object
    # @FIXME: Check for control mesh via a custom property or marker of sorts
    if edit_obj is not None and "Bézier Surface" in edit_obj.name:
        output_object = None
        # @FIXME: Walks all children of control mesh every mesh update
        for child in edit_obj.children:
            if "Bézier Output Object" in child.name:
                output_object = child

        # @TODO: Determine if should spew an error if there is no bezier output
        #   mesh object as a child
        if output_object is not None:
            control_mesh = edit_obj.data

            create_and_replace_output_mesh(control_mesh, output_object)


def add_bezier_surface_button(self, context):
    self.layout.operator(
        OBJECT_OT_add_bezier_surface.bl_idname,
        text="Bézier Surface",
        icon='SURFACE_NSURFACE')


def register():
    """Call to register is made when addon is enabled."""
    bpy.utils.register_class(OBJECT_OT_add_bezier_surface)

    # Detects changes in mesh object in edit mode (8 years ago)
    #   Source: https://gist.github.com/jirihnidek/854beb566c2319affb78
    #   "How should I update a script to 2.8 that uses the removed
    #       'scene_update_pre' handler?"
    #       https://blender.stackexchange.com/q/158164/76575
    bpy.app.handlers.depsgraph_update_pre.append(cb_scene_update)

    # "Extending Menus" Blender API
    #   https://docs.blender.org/api/current/bpy.types.Menu.html#extending-menus
    #   "VIEW3D_MT_surface_add" determined by searching through Blender's Python UI code
    #           Source: https://github.com/blender/blender/blob/482d431bb6735e8206961bd1115d2be7e63572b1/release/scripts/startup/bl_ui/space_view3d.py#L2122
    bpy.types.VIEW3D_MT_surface_add.append(add_bezier_surface_button)


def unregister():
    """Call to unregister is made when addon is disabled."""
    bpy.utils.unregister_class(OBJECT_OT_add_bezier_surface)
    bpy.types.VIEW3D_MT_surface_add.remove(add_bezier_surface_button)

    bpy.app.handlers.depsgraph_update_pre.remove(cb_scene_update)


if __name__ == "__main__":
    register()
