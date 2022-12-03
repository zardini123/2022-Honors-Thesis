"""2022 Honors Thesis Blender Addon."""

# How do I type hint a method with the type of the enclosing class?
#   https://stackoverflow.com/a/33533514
from __future__ import annotations

# Standard imports

import math
import statistics
import typing
import datetime
import dataclasses
import copy
import itertools
import enum

import numpy

# Blender imports

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
if 'math_extensions' in globals():
    importlib.reload(math_extensions)
if 'non_dependent_utilities' in globals():
    importlib.reload(non_dependent_utilities)

from . import utilities
from . import imported_math_from_sympy
from . import math_extensions
from . import non_dependent_utilities

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


def add_value_to_dict(dictionary, key, value_to_add):
    if key not in dictionary:
        dictionary[key] = 0

    dictionary[key] += value_to_add


def append_value_to_dict(dictionary, key, value_to_append):
    if key not in dictionary:
        dictionary[key] = []

    dictionary[key].append(value_to_append)


def add_to_set_value_to_dict(dictionary, key, value_to_append):
    if key not in dictionary:
        dictionary[key] = set()

    dictionary[key].add(value_to_append)


# Dataclass will generate __hash__() when frozen and eq are true
#   https://stackoverflow.com/a/52390731
@dataclasses.dataclass(
    frozen=True,
    eq=True
)
class Zone:
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]
    depth: int
    near_umbilic: bool
    has_ridge: bool
    not_exactly_one_both_ways: bool
    min_deviating: bool
    max_deviating: bool
    min_average_vector: mathutils.Vector
    max_average_vector: mathutils.Vector
    all_magnitudes: float

    def __hash__(self):
        """https://docs.python.org/3/reference/datamodel.html#object.__hash__"""
        return hash((
            self.u_bounds,
            self.v_bounds,
            self.depth,
            self.near_umbilic,
            self.has_ridge,
            self.not_exactly_one_both_ways,
            self.min_deviating,
            self.max_deviating,
            self.min_average_vector.freeze(),
            self.max_average_vector.freeze()
        ))

    def get_width(self) -> float:
        return self.u_bounds[1] - self.u_bounds[0]

    def get_height(self) -> float:
        return self.v_bounds[1] - self.v_bounds[0]

    def get_area(self) -> float:
        return self.get_width() * self.get_height()

    def get_center(self) -> tuple[float, float]:
        return (
            (self.u_bounds[1] + self.u_bounds[0]) / 2,
            (self.v_bounds[1] + self.v_bounds[0]) / 2,
        )

    def distance(self, other_zone: Zone) -> float:
        return math.dist(self.get_center(), other_zone.get_center())

    def center_between_two_zones(self, other_zone: Zone) -> tuple:
        center_1 = self.get_center()
        center_2 = other_zone.get_center()

        return (
            (center_1[0] + center_2[0]) / 2,
            (center_1[1] + center_2[1]) / 2
        )

    @staticmethod
    def parameter_to_zone_index(parameter_val, depth):
        # 4 quadrants per depth
        # 2 splits per axis
        zone_scalar = 2 ** depth

        return int(parameter_val * zone_scalar)

    @staticmethod
    def point_hash(
        point: tuple[float, float],
        depth: int
    ):
        return hash((
            Zone.parameter_to_zone_index(point[0], depth),
            Zone.parameter_to_zone_index(point[1], depth)
        ))

    @staticmethod
    def edge_hash(
        pos_1: tuple[float, float],
        pos_2: tuple[float, float],
        depth: int
    ):
        if pos_2 < pos_1:
            pos_1, pos_2 = pos_2, pos_1

        return hash((
            Zone.point_hash(pos_1, depth),
            Zone.point_hash(pos_2, depth)
        ))

    def get_four_corners(self) -> tuple:
        top_left = (self.u_bounds[0], self.v_bounds[0])
        top_right = (self.u_bounds[1], self.v_bounds[0])
        bottom_left = (self.u_bounds[0], self.v_bounds[1])
        bottom_right = (self.u_bounds[1], self.v_bounds[1])

        return (
            top_left,
            top_right,
            bottom_left,
            bottom_right
        )

    def get_edges_hashes(self) -> tuple:
        top_left, top_right, bottom_left, bottom_right = self.get_four_corners()

        return (
            Zone.edge_hash(top_left, top_right, self.depth),
            Zone.edge_hash(top_left, bottom_left, self.depth),
            Zone.edge_hash(top_right, bottom_right, self.depth),
            Zone.edge_hash(bottom_left, bottom_right, self.depth)
        )

    def get_corner_hashes(self) -> tuple:
        top_left, top_right, bottom_left, bottom_right = self.get_four_corners()

        return (
            Zone.point_hash(top_left, self.depth),
            Zone.point_hash(top_right, self.depth),
            Zone.point_hash(bottom_left, self.depth),
            Zone.point_hash(bottom_right, self.depth)
        )

    def rasterize_to_image_buffer(
        self,
        image_buffer,
        image_size,
        identifier=None,
        do_umbilic_ridges=False,
        color_override=None
    ) -> None:
        # Add to texture
        u_pixel_start = int(round(image_size[0] * self.u_bounds[0]))
        u_pixel_end = int(round(image_size[0] * self.u_bounds[1]))

        v_pixel_start = int(round(image_size[1] * self.v_bounds[0]))
        v_pixel_end = int(round(image_size[1] * self.v_bounds[1]))

        for v_pixel in range(v_pixel_start, v_pixel_end):
            for u_pixel in range(u_pixel_start, u_pixel_end):
                if not do_umbilic_ridges:
                    index = self.depth
                    if identifier is not None:
                        index = identifier

                    color = non_dependent_utilities.unique_color_from_number(index)
                else:
                    color = non_dependent_utilities.deviating_color(
                        self.has_ridge,
                        self.min_deviating,
                        self.max_deviating
                    )

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

                if color_override is not None:
                    color = color_override

                image_buffer[v_pixel][u_pixel] = color


def equal_curvature_zones_recursion(
    bezier_patch_control_points,
    max_depth: int,
    same_direction_magnitude_threshold: float,
    ridge_magnitude_threshold: float,
    near_zero_threshold: float,
    num_samples_per_axis: int,
):
    start_depth = 0

    stack = []
    stack.append(
        (start_depth, (0, 1), (0, 1))
    )

    output_zones = []

    while len(stack) != 0:
        stack_tuple = stack.pop()
        current_depth, u_bounds, v_bounds = stack_tuple

        if current_depth > 1000:
            print("ERROR: MAX STACK DEPTH HIT")
            break

        # Determine if principle direction orientations are generally
        #   in the same orientation

        need_subdivide = False

        min_p_summed_vectors = mathutils.Vector((0, 0))
        max_p_summed_vectors = mathutils.Vector((0, 0))
        count = 0

        # min_principle_directions = set()
        # max_principle_directions = set()

        # min_previous_vector = None
        # max_previous_vector = None

        # min_mag = float('inf')
        # max_mag = float('inf')

        min_mag = 0.0
        max_mag = 0.0

        lowest_ratio = float('inf')

        if num_samples_per_axis == 1:
            u = math_extensions.lerp(u_bounds[0], u_bounds[1], 0.5)
            v = math_extensions.lerp(v_bounds[0], v_bounds[1], 0.5)

            min_principal_vector, max_principal_vector = \
                math_extensions.min_max_principle_directions(
                    bezier_patch_control_points, u, v
                )

            min_p_summed_vectors += min_principal_vector.normalized()
            max_p_summed_vectors += max_principal_vector.normalized()

            count += 1
        else:
            for v_index in range(num_samples_per_axis):
                for u_index in range(num_samples_per_axis):
                    u_pre = u_index / (num_samples_per_axis - 1)
                    v_pre = v_index / (num_samples_per_axis - 1)

                    u = math_extensions.lerp(u_bounds[0], u_bounds[1], u_pre)
                    v = math_extensions.lerp(v_bounds[0], v_bounds[1], v_pre)

                    min_principal_vector, max_principal_vector = \
                        math_extensions.min_max_principle_directions(
                            bezier_patch_control_points, u, v
                        )

                    # if min_mag > min_principal_vector.length and max_mag > max_principal_vector.length:
                    #     min_mag = min_principal_vector.length
                    #     max_mag = max_principal_vector.length

                    min_mag += min_principal_vector.length
                    max_mag += max_principal_vector.length

                    # numerator = min_principal_vector.length
                    # denominator = max_principal_vector.length
                    #
                    # if denominator > numerator:
                    #     numerator, denominator = denominator, numerator
                    #
                    # ratio = numerator / denominator
                    #
                    # if numerator < 1e-10 or denominator < 1e-10:
                    #     ratio = 0.0
                    #
                    # if ratio < lowest_ratio:
                    #     lowest_ratio = ratio

                    if False:
                        world_point = math_extensions.bezier_surface_at_parameters(
                            bezier_patch_control_points, u, v
                        )
                        world_min_principal_vector = math_extensions.bezier_surface_at_parameters(
                            bezier_patch_control_points,
                            u + min_principal_vector.x,
                            v + min_principal_vector.y
                        )
                        world_max_principal_vector = math_extensions.bezier_surface_at_parameters(
                            bezier_patch_control_points,
                            u + max_principal_vector.x,
                            v + max_principal_vector.y
                        )

                        min_principal_vector = world_min_principal_vector - world_point
                        max_principal_vector = world_max_principal_vector - world_point

                    min_p_summed_vectors += min_principal_vector.normalized()
                    max_p_summed_vectors += max_principal_vector.normalized()

                    count += 1

        all_magnitudes = (min_mag + max_mag) / count

        if False:
            assert lowest_ratio != float('inf')

            if lowest_ratio == 0.0:
                all_magnitudes = 0.0
            else:
                all_magnitudes = lowest_ratio - 1

            if lowest_ratio < 1.005:
                print(lowest_ratio, all_magnitudes)

        if False:
            min_mag /= count
            max_mag /= count

            numerator = min_mag
            denominator = max_mag

            if denominator > numerator:
                numerator, denominator = denominator, numerator

            if numerator == 0.0 or denominator == 0.0:
                all_magnitudes = 0.0
            else:
                ratio = numerator / denominator

                all_magnitudes = ratio - 1

                if ratio < 1.005:
                    print(min_mag, max_mag, ratio, all_magnitudes)

        min_p_average_vector = min_p_summed_vectors / count
        max_p_average_vector = max_p_summed_vectors / count

        # print("min:", min_p_average_vector.magnitude)
        # print("max:", max_p_average_vector.magnitude)

        has_ridge = False
        near_umbilic = False
        not_exactly_one_both_ways = False

        min_deviating = False
        max_deviating = False

        # Ridge is opposite direction, therefore results in near 0 magnitude
        if min_p_average_vector.magnitude < ridge_magnitude_threshold:
            if max_p_average_vector.magnitude > same_direction_magnitude_threshold:
                has_ridge = True
                min_deviating = True
        else:
            # If less magnitude (more chaos) in min direction but not enough for being a ridge, near umbilic
            if min_p_average_vector.magnitude < same_direction_magnitude_threshold:
                near_umbilic = True
                min_deviating = True

        if max_p_average_vector.magnitude < ridge_magnitude_threshold:
            if min_p_average_vector.magnitude > same_direction_magnitude_threshold:
                has_ridge = True
                max_deviating = True
        else:
            if max_p_average_vector.magnitude < same_direction_magnitude_threshold:
                near_umbilic = True
                max_deviating = True

        if min_p_average_vector.magnitude < ridge_magnitude_threshold \
                and max_p_average_vector.magnitude < ridge_magnitude_threshold \
                and min_p_average_vector.magnitude > near_zero_threshold \
                and max_p_average_vector.magnitude > near_zero_threshold:
            not_exactly_one_both_ways = True

        if has_ridge or near_umbilic or not_exactly_one_both_ways:
            if current_depth < max_depth:
                need_subdivide = True
            # else:
            #     current_depth += 1

        if not need_subdivide:
            output_zones.append(
                Zone(
                    u_bounds,
                    v_bounds,
                    current_depth,
                    near_umbilic,
                    has_ridge,
                    not_exactly_one_both_ways,
                    min_deviating,
                    max_deviating,
                    min_p_average_vector,
                    max_p_average_vector,
                    all_magnitudes
                )
            )
            continue

        next_depth = current_depth + 1
        half_u = (u_bounds[0] + u_bounds[1]) / 2
        half_v = (v_bounds[0] + v_bounds[1]) / 2

        # Bottom left
        stack.append(
            (next_depth, (u_bounds[0], half_u), (v_bounds[0], half_v))
        )
        # Bottom right
        stack.append(
            (next_depth, (half_u, u_bounds[1]), (v_bounds[0], half_v))
        )
        # Top Left
        stack.append(
            (next_depth, (u_bounds[0], half_u), (half_v, v_bounds[1]))
        )
        # Top Right
        stack.append(
            (next_depth, (half_u, u_bounds[1]), (half_v, v_bounds[1]))
        )

    return output_zones


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
        # Image
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

            # Create grid mesh
            if True:
                uv_layer = output_bmesh.loops.layers.uv.verify()

                for patch_index, patch_control_points in enumerate(patches):
                    def generator_function(u, v):
                        u_bound = (0, 1)
                        v_bound = (0, 1)

                        return math_extensions.bezier_surface_at_parameters(
                            patch_control_points,
                            math_extensions.lerp(u_bound[0], u_bound[1], u),
                            math_extensions.lerp(v_bound[0], v_bound[1], v)
                        )

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

            # Setup material and image data blocks
            if True:
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
                            try:
                                index_str = image_name.replace(patch_prefix, "")
                                index = int(index_str)
                                images[index] = image
                            except ValueError:
                                pass

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
                        link = links.new(
                            node_texture.outputs["Color"], node_diffuse.inputs["Color"])
                        link = links.new(
                            node_diffuse.outputs["BSDF"], node_output.inputs["Surface"])

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

            # Create image
            if True:
                image_buffer = numpy.zeros((v_image_size, u_image_size, 4))

                u_num_sample_points = 15
                v_num_sample_points = 15

                u_scan_size = 10
                v_scan_size = 10

                # Modify textures to display surface values
                for patch_index, image in enumerate(images):
                    patch = patches[patch_index]

                    num_data_points = len(image.pixels)
                    u_size, v_size = image.size
                    num_channels = image.channels

                    # Recursive method
                    if True:
                        draw_zones_for_all = True
                        only_draw_base_zones = True

                        show_curvature_lines = False
                        only_show_base_curvature = False
                        draw_ajacent_ridges = False

                        print("getting zones")

                        # max_depth = 10
                        # same_direction_magnitude_threshold = 0.999
                        # ridge_magnitude_threshold = 0.6
                        # near_zero_threshold = 0.01
                        # # Need 3 because of no umbilics, many ridges saddle
                        # num_samples_per_axis = 3

                        show_ridges = True

                        max_depth = 7
                        same_direction_magnitude_threshold = 0.999
                        ridge_magnitude_threshold = 0.6
                        near_zero_threshold = 0.01
                        num_samples_per_axis = 10

                        zones = equal_curvature_zones_recursion(
                            patch,
                            max_depth,
                            same_direction_magnitude_threshold,
                            ridge_magnitude_threshold,
                            near_zero_threshold,
                            num_samples_per_axis,
                        )

                        print("num zones:", len(zones))

                        area_percentages = {}

                        for zone in zones:
                            add_value_to_dict(area_percentages, zone.depth, zone.get_area())

                            # val = max(0, min(1, zone.all_magnitudes))

                            if draw_zones_for_all:
                                zone.rasterize_to_image_buffer(
                                    image_buffer, image.size,
                                    do_umbilic_ridges=show_ridges,
                                    # color_override=(val, val, val, 1.0)
                                )

                        if not only_draw_base_zones:
                            print(dict(sorted(area_percentages.items(), key=lambda item: item[1])))
                            print("area sum:", sum(area_percentages.values()))

                            if len(area_percentages) >= 2:
                                print(statistics.mean(area_percentages.values()))
                                print(statistics.stdev(area_percentages.values()))

                            print("getting zone shared edges")

                            # Maps edge hash to zones with edge
                            final_depth_zone_shared_edge = {}
                            umbilic_zones = set()

                            for zone in zones:
                                if zone.depth == max_depth:
                                    if zone.near_umbilic:
                                        umbilic_zones.add(zone)

                                    edge_hashes = zone.get_edges_hashes()
                                    for edge_hash in edge_hashes:
                                        append_value_to_dict(
                                            final_depth_zone_shared_edge, edge_hash, zone)

                            print("num umbilic zones:", len(umbilic_zones))

                            print("verifying shared edge invariants")
                            for zones_with_edge in final_depth_zone_shared_edge.values():
                                assert len(zones_with_edge) <= 4

                                # if len(zones_with_corner) == 2:
                                #     dist = (
                                #         mathutils.Vector(zones_with_corner[0].get_center())
                                #         - mathutils.Vector(zones_with_corner[1].get_center())
                                #     ).length
                                #     # Check against distance between max depth zones
                                #     #   (which is width of a max depth zone)
                                #     assert abs(dist - 1 / (2 ** max_depth)) <= 1e-6

                            print("create umbilic/ridge adjacency graph")

                            # Maps zone to zones with shared edges
                            final_depth_adjacency_graph = {}

                            for edge_hash, zones in final_depth_zone_shared_edge.items():
                                for pair in itertools.permutations(zones, r=2):
                                    add_to_set_value_to_dict(
                                        final_depth_adjacency_graph, pair[0], pair[1]
                                    )
                                    add_to_set_value_to_dict(
                                        final_depth_adjacency_graph, pair[1], pair[0]
                                    )

                            print("verifying adjacency invariants")
                            for start_zone, adjacency in final_depth_adjacency_graph.items():
                                assert len(adjacency) <= 8 and len(adjacency) > 0

                                # Assert adjacency does not map onto itself
                                for zone in adjacency:
                                    # Check if not same location in memory
                                    assert zone is not start_zone

                            print("get each umbilic area")

                            unvisited_umbilic_zones = copy.copy(umbilic_zones)
                            umbilic_areas = []

                            while len(unvisited_umbilic_zones) != 0:
                                start_umbilic_zone = unvisited_umbilic_zones.pop()

                                umbilic_area = {
                                    "zones": set(),
                                    "min_deviating": set(),
                                    "max_deviating": set(),
                                    "centers": set()
                                }

                                current_umbilic_area_queue = set()
                                current_umbilic_area_queue.add(start_umbilic_zone)

                                min_deviating = set()
                                max_deviating = set()

                                total_magnitude = 0.0
                                smallest_magnitude = float("inf")
                                current_smallest_zone = None
                                count = 0

                                while len(current_umbilic_area_queue) != 0:
                                    current_umbilic_zone = current_umbilic_area_queue.pop()

                                    umbilic_area["zones"].add(current_umbilic_zone)

                                    count += 1

                                    # if current_umbilic_zone.near_umbilic:
                                    #     total_magnitude += current_umbilic_zone.all_magnitudes
                                    #
                                    #     if current_umbilic_zone.all_magnitudes < smallest_magnitude:
                                    #         smallest_magnitude = current_umbilic_zone.all_magnitudes
                                    #         current_smallest_zone = current_umbilic_zone

                                    if current_umbilic_zone != start_umbilic_zone:
                                        unvisited_umbilic_zones.remove(current_umbilic_zone)

                                    for adjacent_zone in final_depth_adjacency_graph[current_umbilic_zone]:
                                        # Can have zones without ridge or umbilic
                                        if adjacent_zone.has_ridge:
                                            if adjacent_zone.min_deviating:
                                                umbilic_area["min_deviating"].add(adjacent_zone)
                                            elif adjacent_zone.max_deviating:
                                                umbilic_area["max_deviating"].add(adjacent_zone)

                                        elif adjacent_zone.near_umbilic:
                                            if adjacent_zone in unvisited_umbilic_zones:
                                                current_umbilic_area_queue.add(adjacent_zone)

                                # assert current_smallest_zone is not None
                                # assert smallest_magnitude != float("inf")
                                #
                                # average_magnitude = total_magnitude / count
                                # print(count, smallest_magnitude)
                                # print(average_magnitude)

                                # if count > 1
                                # current_smallest_zone.rasterize_to_image_buffer(
                                #     image_buffer, image.size,
                                #     color_override=(1.0, 1.0, 1.0, 1.0)
                                # )

                                # center = current_smallest_zone.get_center()
                                # utilities.rasterize_point(
                                #     image_buffer,
                                #     mathutils.Vector(center),
                                #     image.size,
                                #     (1.0, 1.0, 1.0, 1.0)
                                # )

                                if len(umbilic_area["zones"]) <= 1:
                                    continue

                                umbilic_areas.append(umbilic_area)

                            print("num umbilic areas:", len(umbilic_areas))

                            umbilic_centers = set()

                            index = 5

                            for umbilic_area in umbilic_areas:
                                # print("AAAAAAA")

                                # for zone in umbilic_area["zones"]:
                                #     print(zone.all_magnitudes)

                                # for current in current_umbilic_zones:
                                #     current.rasterize_to_image_buffer(
                                #         image_buffer, image.size, index
                                #     )

                                min_val = min(zone.all_magnitudes for zone in umbilic_area["zones"])
                                print(min_val)

                                centers = set()

                                if min_val < 1:
                                    current_umbilic_zones = set()

                                    for zone in umbilic_area["zones"]:
                                        queue = set()
                                        visited = set()

                                        queue.add(zone)
                                        while len(queue) != 0:
                                            current_zone = queue.pop()

                                            if current_zone in visited:
                                                continue

                                            visited.add(current_zone)

                                            none_lower = True
                                            for adjacent_zone in final_depth_adjacency_graph[current_zone]:
                                                if adjacent_zone.all_magnitudes < zone.all_magnitudes:
                                                    queue.add(adjacent_zone)
                                                    none_lower = False

                                            if none_lower:
                                                current_umbilic_zones.add(current_zone)

                                    for current in current_umbilic_zones:
                                        current.rasterize_to_image_buffer(
                                            image_buffer, image.size, index
                                        )

                                    print("current zones", len(current_umbilic_zones))

                                    bins = {}

                                    # for current in current_umbilic_zones:
                                    #     centers.add(mathutils.Vector(current.get_center()).freeze())

                                    if True:
                                        for current in current_umbilic_zones:
                                            is_similar = False
                                            for comparison in bins.keys():
                                                zone_1 = mathutils.Vector(current.get_center())
                                                zone_2 = mathutils.Vector(comparison.get_center())
                                                distance = (zone_1 - zone_2).length

                                                if distance < (1 / 2 ** 6):
                                                    is_similar = True
                                                    break

                                            if is_similar:
                                                append_value_to_dict(bins, comparison, current)
                                            else:
                                                bins[current] = [current]

                                        print("bins", len(bins))

                                        for zones in bins.values():
                                            final_center = mathutils.Vector((0, 0))
                                            for zone in zones:
                                                final_center += mathutils.Vector(zone.get_center())

                                            final_center /= len(zones)

                                            centers.add(final_center.freeze())
                                else:
                                    if len(umbilic_area["min_deviating"]) == 0 \
                                            or len(umbilic_area["max_deviating"]) == 0:
                                        continue

                                    if draw_ajacent_ridges:
                                        for min_deviating in umbilic_area["min_deviating"]:
                                            min_deviating.rasterize_to_image_buffer(
                                                image_buffer, image.size, index)
                                        for max_deviating in umbilic_area["max_deviating"]:
                                            max_deviating.rasterize_to_image_buffer(
                                                image_buffer, image.size, index + 1)

                                    print("find closest pair of min and max deviating (brute force)")

                                    # Assumes only 1 min and 1 max ridge

                                    # https://docs.python.org/3/library/itertools.html#itertools.product
                                    min_dist = float('inf')
                                    min_pair = None
                                    for pair in itertools.product(umbilic_area["min_deviating"], umbilic_area["max_deviating"]):
                                        min_zone, max_zone = pair
                                        current_dist = min_zone.distance(max_zone)

                                        # centers.add(
                                        #     mathutils.Vector(
                                        #         min_zone.center_between_two_zones(max_zone)
                                        #     ).freeze()
                                        # )

                                        if current_dist <= min_dist:
                                            min_dist = current_dist
                                            min_pair = pair

                                    assert min_pair is not None

                                    center = mathutils.Vector(
                                        min_pair[0].center_between_two_zones(min_pair[1])
                                    )

                                    centers.add(center.freeze())

                                for center in centers:
                                    utilities.rasterize_point(
                                        image_buffer,
                                        center,
                                        image.size,
                                        (1.0, 1.0, 1.0, 1.0)
                                    )

                                umbilic_area["centers"] = centers
                                umbilic_centers = umbilic_centers.union(centers)

                                index += 2

                            print("num umbilics centers:", len(umbilic_centers))

                        if show_curvature_lines:
                            # # https://stackoverflow.com/a/9997374
                            # def ccw(A, B, C):
                            #     return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
                            #
                            # # Return true if line segments AB and CD intersect
                            # def intersect(A, B, C, D):
                            #     return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

                            class Polyline:
                                def __init__(self,
                                             position_array: list[mathutils.Vector],
                                             length_array: list[float],
                                             min_else_max: bool
                                             ):
                                    self._position_array = position_array
                                    first_val = length_array[0]
                                    self._length_array = [e - first_val for e in length_array]
                                    self._min_else_max = min_else_max

                                def is_min_else_max(self):
                                    return self._min_else_max

                                def get_length(self):
                                    return self._length_array[-1]

                                def coordinate_at_parameter(self, parameter):
                                    length_along = parameter * self.get_length()

                                    print(f"apply polyline split at {parameter}")

                                    index = 0
                                    for pair in itertools.pairwise(self._length_array):
                                        first, second = pair

                                        if length_along >= first and length_along <= second:
                                            scale = second - first
                                            interpolate = (length_along - first) / scale

                                            pos_1 = self._position_array[index]
                                            pos_2 = self._position_array[index + 1]

                                            return pos_1.lerp(pos_2, interpolate)

                                        index += 1

                                    return None

                                def get_intersection(
                                    self,
                                    segment_start_coord: mathutils.Vector,
                                    segment_end_coord: mathutils.Vector
                                ):
                                    index = 0

                                    # https://docs.python.org/3/library/itertools.html#itertools.pairwise
                                    for pair in itertools.pairwise(self._position_array):
                                        first, second = pair

                                        length_along = self._length_array[index]
                                        length_along_next = self._length_array[index + 1]
                                        change_between = length_along_next - length_along

                                        intersection = mathutils.geometry.intersect_line_line_2d(
                                            first,
                                            second,
                                            segment_start_coord,
                                            segment_end_coord
                                        )

                                        if intersection is not None:
                                            delta = mathutils.Vector(
                                                second) - mathutils.Vector(first)
                                            intersection_delta = mathutils.Vector(
                                                intersection) - mathutils.Vector(first)

                                            interpolate = intersection_delta.length / delta.length

                                            parameter = (length_along + (change_between *
                                                         interpolate)) / self.get_length()

                                            return intersection, parameter

                                        index += 1

                                    return None

                                def split_at(self, parameter):
                                    length_along = parameter * self.get_length()

                                    original_length = self.get_length()

                                    print(f"apply polyline split at {parameter}")

                                    index = 0
                                    for pair in itertools.pairwise(self._length_array):
                                        first, second = pair

                                        if length_along == first:
                                            new_polyline = Polyline(
                                                self._position_array[index:],
                                                self._length_array[index:],
                                                self._min_else_max
                                            )

                                            # Modify array afterwards so new polyline can source from it
                                            self._position_array = self._position_array[:index + 1]
                                            self._length_array = self._length_array[:index + 1]

                                            # @TODO: Length assertion
                                            # assert self.get_length() - se

                                            return self, new_polyline

                                        if length_along > first and length_along < second:
                                            scale = second - first
                                            interpolate = (length_along - first) / scale

                                            pos_1 = self._position_array[index]
                                            pos_2 = self._position_array[index + 1]

                                            new_pos = pos_1.lerp(pos_2, interpolate)

                                            # Does not include first position as index + 1
                                            new_polyline = Polyline(
                                                [new_pos] + self._position_array[index + 1:],
                                                [length_along] + self._length_array[index + 1:],
                                                self._min_else_max
                                            )

                                            # Includes first position as index (index + 1 exclusive)
                                            self._position_array = \
                                                self._position_array[:index + 1] + [new_pos]
                                            self._length_array = \
                                                self._length_array[:index + 1] + [length_along]

                                            return self, new_polyline

                                        index += 1

                                    return None

                                def rasterize(self, index=None):
                                    for position in self._position_array:
                                        if index is None:
                                            color = (0.0, 0.0, 1.0, 1.0) if self._min_else_max else (
                                                1.0, 0.0, 0.0, 1.0)
                                        else:
                                            color = non_dependent_utilities.unique_color_from_number(
                                                index)

                                        utilities.rasterize_point(
                                            image_buffer,
                                            position,
                                            image.size,
                                            color
                                        )

                                    pass

                            # Integration (
                            #   vector_field_function (arc length scaled),
                            #   origin_vertex,
                            #   start_position,
                            #   reverse_integration,
                            #   polylines: position_collection[Polyline],
                            #   mesh_verticies: position_collection[Mesh.Vertex]
                            # )
                            # Returns: (
                            #   new_polyline
                            #   ending_vertex (existing if hit_vertex else new)
                            #   result_type,
                            #   intersected_polyline_results = (
                            #       intersected polyline (None if using existing vertex),
                            #       polyline intersection parameter (0 to 1 inclusive),
                            #   )
                            # )

                            class ResultType(enum.Enum):
                                HIT_VERTEX = 1
                                HIT_BOUNDS = 2
                                HIT_POLYLINE = 3
                                RAN_OUT_OF_STEPS = 4

                            def integrate(
                                bezier_patch_control_points,
                                min_else_max: bool,
                                origin_vertex: bmesh.types.BMVert,
                                start_parameters: mathutils.Vector,
                                reverse_integration: bool,
                                polylines: set[Polyline],
                                mesh_verticies: bmesh.types.BMVertSeq
                            ):
                                max_steps = 100000
                                same_pos_tolerance = 1e-9
                                same_vertex_distance = 1 / (2 ** 7)
                                step_size = 0.001

                                position_array = []
                                length_array = []

                                origin_parameters = origin_vertex.co.to_2d().freeze()
                                vertex_dist = (origin_parameters - start_parameters).length

                                if vertex_dist > same_pos_tolerance:
                                    position_array = [origin_parameters,
                                                      start_parameters.freeze()]
                                    length_array = [0.0, vertex_dist]

                                    current_length = vertex_dist
                                else:
                                    position_array = [start_parameters.freeze()]
                                    length_array = [0.0]

                                    current_length = 0.0

                                current_principle_vector = None
                                current_parameters = mathutils.Vector(start_parameters)

                                result_type = ResultType.RAN_OUT_OF_STEPS
                                intersection_return = None
                                ending_vertex = None

                                for step in range(max_steps):
                                    val_in_bounds, new_principle_vector, new_parameters, reverse_due_to_ridge = \
                                        math_extensions.line_of_curvature_integration_fixed_step_01_bound(
                                            bezier_patch_control_points,
                                            current_parameters,
                                            min_else_max,
                                            step_size,
                                            last_principle_vector=current_principle_vector,
                                            reverse=reverse_integration
                                        )

                                    if reverse_due_to_ridge:
                                        reverse_integration = not reverse_integration

                                    position_array.append(new_parameters.freeze())

                                    current_length += step_size
                                    length_array.append(current_length)

                                    if not val_in_bounds:
                                        result_type = ResultType.HIT_BOUNDS
                                        break

                                    # distance_from_origin = (origin_parameters - new_parameters).length

                                    if current_length > step_size * 2:
                                        # Find nearby verticies
                                        for vert_to_compare in mesh_verticies:
                                            # Ignore origin vertex
                                            if vert_to_compare is origin_vertex:
                                                continue

                                            if (vert_to_compare.co.to_2d() - new_parameters).length <= same_vertex_distance:
                                                result_type = ResultType.HIT_VERTEX
                                                ending_vertex = vert_to_compare
                                                break

                                        # Find intersecting polylines
                                        for polyline in polylines:
                                            # if polyline.is_min_else_max() == min_else_max:
                                            #     continue

                                            intersection_results = polyline.get_intersection(
                                                current_parameters,
                                                new_parameters,
                                            )

                                            if intersection_results is not None:
                                                point, parameter = intersection_results

                                                if parameter == 0.0:
                                                    continue

                                                result_type = ResultType.HIT_POLYLINE

                                                position_array[-1] = point
                                                length_array[-1] -= step_size * ((new_parameters - point).length /
                                                                                 (new_parameters - current_parameters).length)

                                                print(f"polyline hit at {point}")

                                                intersection_return = (
                                                    polyline,
                                                    parameter
                                                )
                                                break

                                    if result_type != ResultType.RAN_OUT_OF_STEPS:
                                        break

                                    current_parameters = new_parameters
                                    current_principle_vector = new_principle_vector

                                if ending_vertex is None:
                                    ending_vertex = mesh_verticies.new(
                                        position_array[-1].to_3d()
                                    )

                                new_polyline = Polyline(
                                    position_array,
                                    length_array,
                                    min_else_max
                                )

                                # Returns: (
                                #   new_polyline
                                #   ending_vertex (existing if hit_vertex else new)
                                #   result_type,
                                #   intersected_polyline_results (None if using existing vertex):
                                #   (
                                #       intersected polyline,
                                #       polyline intersection parameter (0 to 1 inclusive),
                                #   )
                                # )
                                return (
                                    new_polyline,
                                    ending_vertex,
                                    result_type,
                                    reverse_integration,
                                    intersection_return
                                )

                            # Cut Polyline and Edge and Add Vertex (
                            #   polyline_to_edge_map: dict[Polyline, Mesh.Edge],
                            #   mesh_edges_collection: collection[Mesh.Edge]
                            #   polyline_to_split: Polyline,
                            #   parameter: float,
                            #   vertex_to_add: Mesh.Vertex
                            # )
                            #

                            def cut_polyline_and_edge_and_add_vertex(
                                polylines: set[Polyline],
                                polyline_to_edge_map: dict[Polyline, bmesh.types.BMEdge],
                                mesh_edges_collection: bmesh.types.BMEdgeSeq,
                                polyline_to_split: Polyline,
                                parameter: float,
                                vertex_to_add: bmesh.types.BMVert
                            ):
                                print("apply cut")

                                edge_to_split = polyline_to_edge_map[polyline_to_split]

                                # Make two new edges
                                start_vert, end_vert = edge_to_split.verts

                                edge_lower = mesh_edges_collection.get((start_vert, vertex_to_add))
                                edge_upper = mesh_edges_collection.get((vertex_to_add, end_vert))

                                if edge_lower is not None or edge_upper is not None:
                                    return

                                # Remove all traces of original polyline
                                polyline_to_edge_map.pop(polyline_to_split)
                                polylines.remove(polyline_to_split)

                                polyline_lower, polyline_upper = polyline_to_split.split_at(
                                    parameter)

                                # for edge in mesh_edges_collection:
                                #     print(edge)

                                mesh_edges_collection.remove(edge_to_split)

                                print("start:", start_vert)
                                print("vertex_to_add:", vertex_to_add)
                                print("end:", end_vert)

                                edge_lower = mesh_edges_collection.new((start_vert, vertex_to_add))
                                edge_upper = mesh_edges_collection.new((vertex_to_add, end_vert))

                                print("lower:", edge_lower)
                                print("upper:", edge_upper)

                                # Add two parts of original polyline
                                polylines.add(polyline_lower)
                                polylines.add(polyline_upper)

                                # polyline_lower.rasterize(index=id(polyline_upper))
                                # polyline_upper.rasterize(index=id(polyline_upper))

                                polyline_to_edge_map[polyline_lower] = edge_lower
                                polyline_to_edge_map[polyline_upper] = edge_upper

                            # Create line of curvature edge (
                            #   function_to_integrate
                            #   starting_vertex
                            #   starting_point,
                            #   reverse_integration: bool,
                            #   polyline_to_edge_map: dict[Polyline, Mesh.Edge]
                            #   polylines: position_collection[Polyline],
                            #   mesh_verticies: position_collection[Mesh.Vertex]
                            #   mesh_edges: collection[Mesh.Edges]
                            # )
                            #
                            # Returns: (
                            #   lonely_vertex, None when hit_vertex or hit_bounds
                            # )

                            def create_line_of_curvature_edge(
                                bezier_patch_control_points,
                                min_else_max,
                                starting_vertex: bmesh.types.BMVert,
                                start_position: mathutils.Vector,
                                reverse_integration: bool,
                                polyline_to_edge_map: dict[Polyline, bmesh.types.BMEdge],
                                polylines: set[Polyline],
                                mesh_verticies: bmesh.types.BMVertSeq,
                                mesh_edges: bmesh.types.BMEdgeSeq
                            ):
                                print(f"starting vertex {starting_vertex}")
                                print(f"starting vertex at {starting_vertex.co}")
                                print(f"starting at {start_position}")

                                new_polyline, ending_vertex, result_type, final_reversed_state, intersected_polyline_results = integrate(
                                    bezier_patch_control_points,
                                    min_else_max,
                                    starting_vertex,
                                    start_position,
                                    reverse_integration,
                                    polylines,
                                    mesh_verticies
                                )

                                assert result_type != ResultType.RAN_OUT_OF_STEPS

                                print(f"ending vertex {ending_vertex}")
                                print(f"ending vertex at {ending_vertex.co}")

                                assert starting_vertex is not ending_vertex

                                print(result_type)

                                old_edge = mesh_edges.get((starting_vertex, ending_vertex))
                                if old_edge is not None:
                                    return None

                                polylines.add(new_polyline)

                                new_edge = mesh_edges.new((starting_vertex, ending_vertex))
                                polyline_to_edge_map[new_polyline] = new_edge

                                if result_type == ResultType.HIT_POLYLINE:
                                    intersected_polyline, polyline_intersection_parameter = intersected_polyline_results

                                    assert new_polyline is not intersected_polyline
                                    # New vert, must only be connected to edge just made
                                    assert len(ending_vertex.link_edges) == 1

                                    print("polyline intersect vertex:", ending_vertex)

                                    cut_polyline_and_edge_and_add_vertex(
                                        polylines,
                                        polyline_to_edge_map,
                                        mesh_edges,
                                        intersected_polyline,
                                        polyline_intersection_parameter,
                                        ending_vertex,
                                    )

                                    print("polyline intersect vertex:", ending_vertex)

                                    return (ending_vertex, final_reversed_state)

                                return None

                            #   secondary_ridges = get_secondary_ridges_starting_points(
                            #       min_principle_direction,
                            #       max_principle_direction,
                            #       umbilic_center,
                            #       scanning_radius = 1 / (2 ** 9)
                            #   )

                            def get_secondary_ridges_starting_points(
                                bezier_patch_control_points,
                                center: mathutils.Vector,
                                scanning_radius: float,
                                subdivisions: int,
                                starting_dot_tolerance: float
                            ) -> list[tuple[mathutils.Vector, bool, bool]]:

                                is_in_extreme = False
                                last_is_in_extreme = False
                                min_is_most_extreme = False
                                reverse_integration = False

                                extreme_dot = 0.0
                                extreme_point = None

                                output = []

                                for subdivision in range(subdivisions):
                                    parameter = (subdivision / subdivisions)
                                    t = parameter * 2 * math.pi

                                    change = mathutils.Vector((
                                        math.cos(t),
                                        math.sin(t)
                                    ))

                                    final_point = center + (change * scanning_radius)

                                    min_principal_vector, max_principal_vector = \
                                        math_extensions.min_max_principle_directions(
                                            bezier_patch_control_points,
                                            final_point.x,
                                            final_point.y
                                        )

                                    min_dot = min_principal_vector.normalized().dot(change.normalized())
                                    max_dot = max_principal_vector.normalized().dot(change.normalized())

                                    if abs(min_dot) > starting_dot_tolerance:
                                        is_in_extreme = True

                                        reverse_integration = min_dot < 0

                                        if abs(min_dot) > extreme_dot:
                                            extreme_point = final_point.freeze()
                                            extreme_dot = abs(min_dot)

                                        min_is_most_extreme = True
                                    elif abs(max_dot) > starting_dot_tolerance:
                                        is_in_extreme = True

                                        reverse_integration = max_dot < 0

                                        if abs(max_dot) > extreme_dot:
                                            extreme_point = final_point.freeze()
                                            extreme_dot = abs(max_dot)

                                        min_is_most_extreme = False
                                    else:
                                        is_in_extreme = False

                                    if last_is_in_extreme and not is_in_extreme:
                                        if extreme_point is not None:
                                            print(min_is_most_extreme,
                                                  reverse_integration, min_dot, max_dot)

                                            output.append((
                                                extreme_point, min_is_most_extreme, reverse_integration
                                            ))

                                            # order.append("B" if min_is_most_extreme else "R")
                                            #
                                            color = (0.0, 0.0, 1.0, 1.0) if min_is_most_extreme else (
                                                1.0, 0.0, 0.0, 1.0)

                                            utilities.rasterize_point(
                                                image_buffer,
                                                extreme_point,
                                                image.size,
                                                color
                                            )

                                            #
                                            # math_extensions.integrate_lines_of_curvature_image(
                                            #     image_buffer,
                                            #     image.size,
                                            #     color,
                                            #     patch,
                                            #     extreme_point,
                                            #     arc_length_distance,
                                            #     step_size,
                                            #     use_min_principle_direction=min_is_most_extreme,
                                            #     reverse=reverse
                                            # )

                                            print(extreme_dot)

                                            extreme_dot = 0.0
                                            extreme_point = None

                                    last_is_in_extreme = is_in_extreme

                                return output

                            print("begin meshing")

                            # Following will be queried for nearby components:
                            # output_mesh = Mesh()
                            #   where
                            #       output_mesh.verts is a collection of Vertex,
                            #           where Vertex contains at minimum two dimensions for coordinate
                            #       output_mesh.edges is a collection of Edge,
                            #           where Edge contains two Vertex defining the edge

                            polylines: set[Polyline] = set()

                            polyline_to_edge_map: dict[Polyline, bmesh.types.BMEdge] = {}
                            lonely_verticies: list[tuple[bmesh.types.BMVert, bool, bool]] = []

                            print("setup generator points")

                            if len(umbilic_centers) > 0:
                                for umbilic_center in umbilic_centers:
                                    umbilic_center = mathutils.Vector(umbilic_center)

                                    umbilic_center_vertex = output_bmesh.verts.new(
                                        umbilic_center.to_3d()
                                    )

                                    print("umbilic vertex:", umbilic_center_vertex)

                                    # : list[tuple[starting_point, min_else_max, reverse]]
                                    secondary_ridges = get_secondary_ridges_starting_points(
                                        patch,
                                        umbilic_center,
                                        scanning_radius=1 / (2 ** 9),
                                        subdivisions=1000,
                                        starting_dot_tolerance=0.999,
                                    )

                                    for secondary_ridge in secondary_ridges:
                                        integration_starting_point, min_else_max, reverse_integration = secondary_ridge

                                        lonely_verticies.append((
                                            False, umbilic_center_vertex, integration_starting_point, min_else_max, reverse_integration
                                        ))
                            else:
                                center_pos = mathutils.Vector((0.5, 0.5))
                                center_vertex = output_bmesh.verts.new(
                                    center_pos.to_3d()
                                )

                                for pair in itertools.product([True, False], repeat=2):
                                    lonely_verticies.append((
                                        False, center_vertex, center_pos, pair[0], pair[1]
                                    ))

                            while True:
                                while len(lonely_verticies) != 0:
                                    print(len(lonely_verticies))
                                    lonely_tuple = lonely_verticies.pop(0)
                                    is_lonely, lonely_vertex, integration_starting_point, min_else_max, reverse_integration = lonely_tuple

                                    print(lonely_tuple)

                                    lonely_vertex_result = create_line_of_curvature_edge(
                                        patch,
                                        min_else_max,
                                        lonely_vertex,
                                        integration_starting_point,
                                        reverse_integration,
                                        polyline_to_edge_map,
                                        polylines,
                                        output_bmesh.verts,
                                        output_bmesh.edges
                                    )

                                    if lonely_vertex_result is not None:
                                        new_lonely_vertex, final_reversed_state = lonely_vertex_result

                                        print(f"new lonely vertex: {new_lonely_vertex}")
                                        print(f"new lonely vertex at {new_lonely_vertex.co}")

                                        # Continue in same direction as previous lonely direction
                                        start_coordinate = new_lonely_vertex.co.to_2d().freeze()
                                        lonely_verticies.append((
                                            True, new_lonely_vertex, start_coordinate, min_else_max, final_reversed_state
                                        ))

                                # No lonely verticies here

                                print("Finding edges to subdivide")

                                for polyline in polylines:
                                    print(polyline.get_length())

                                if only_show_base_curvature:
                                    break

                                # Find first polyline that needs to be subdivided
                                polyline_to_split = None
                                for polyline in polylines:
                                    if polyline.get_length() >= 1 / (2 ** 2):
                                        polyline_to_split = polyline
                                        break

                                # No more work to be done
                                if polyline_to_split is None:
                                    break

                                halfway_coord = polyline_to_split.coordinate_at_parameter(
                                    0.5).freeze()

                                assert halfway_coord is not None

                                halfway_vertex = output_bmesh.verts.new(
                                    halfway_coord.to_3d()
                                )

                                cut_polyline_and_edge_and_add_vertex(
                                    polylines,
                                    polyline_to_edge_map,
                                    output_bmesh.edges,
                                    polyline_to_split,
                                    0.5,
                                    halfway_vertex
                                )

                                # Flip integration function for orthogonality to edge
                                min_else_max = not polyline_to_split.is_min_else_max()
                                lonely_verticies.append((
                                    True, halfway_vertex, halfway_coord, min_else_max, False
                                ))
                                lonely_verticies.append((
                                    True, halfway_vertex, halfway_coord, min_else_max, True
                                ))

                            enable_boundary = True
                            if enable_boundary:
                                print("creating boundary edges")

                                bottom_left = mathutils.Vector((0, 0))
                                bottom_right = mathutils.Vector((1, 0))

                                top_left = mathutils.Vector((0, 1))
                                top_right = mathutils.Vector((1, 1))

                                corners_to_consider = [
                                    bottom_left.freeze(),
                                    bottom_right.freeze(),
                                    top_left.freeze(),
                                    top_right.freeze()
                                ]

                                def in_tolerance(val_1, val_2, epsilon):
                                    return abs(val_1 - val_2) <= epsilon

                                def to_index(val_1, val_2):
                                    return (int(val_1), int(val_2))

                                corners = {}

                                for vert in output_bmesh.verts:
                                    for corner in corners_to_consider:
                                        if in_tolerance(vert.co.x, corner.x, 1/(2**9)) and in_tolerance(vert.co.y, corner.y, 1/(2**9)):
                                            append_value_to_dict(
                                                corners, to_index(corner.x, corner.y), vert
                                            )
                                            break

                                print("corners", corners)

                                for pair in itertools.product([0, 1], repeat=2):
                                    index = to_index(pair[0], pair[1])
                                    if index not in corners:
                                        corners[index] = output_bmesh.verts.new(
                                            mathutils.Vector((pair[0], pair[1], 0.0))
                                        )

                                edges = {}

                                for vert in output_bmesh.verts:
                                    left = in_tolerance(vert.co.x, 0, 1/(2**9))
                                    right = in_tolerance(vert.co.x, 1, 1/(2**9))
                                    up = in_tolerance(vert.co.y, 1, 1/(2**9))
                                    down = in_tolerance(vert.co.y, 0, 1/(2**9))

                                    if left:
                                        sort_val = vert.co.y
                                        append_value_to_dict(edges, 0, (sort_val, vert))
                                    if right:
                                        sort_val = vert.co.y
                                        append_value_to_dict(edges, 1, (sort_val, vert))
                                    if up:
                                        sort_val = vert.co.x
                                        append_value_to_dict(edges, 2, (sort_val, vert))
                                    if down:
                                        sort_val = vert.co.x
                                        append_value_to_dict(edges, 3, (sort_val, vert))

                                print('edges', edges)

                                for edge in edges.values():
                                    edge.sort(key=lambda x: x[0])

                                    for pair in itertools.pairwise(edge):
                                        vert_1 = pair[0][1]
                                        vert_2 = pair[1][1]
                                        output_bmesh.edges.new(
                                            (vert_1, vert_2)
                                        )

                            print("finished meshing")

                            print("rasterizing polylines")

                            for polyline in polylines:
                                polyline.rasterize()

                        if False:
                            print("get areas of same type and their adjacency")

                            if len(special_final_depth_zones) > 0:
                                seperate_from_current_area = {
                                    zone: None
                                    for zone in special_final_depth_zones
                                }

                                visited_zones = set()

                                # Maps area id to adjacent areas
                                area_adjacency = {}
                                # Maps area id to area zones
                                area_id_map = {}
                                ridge_areas = set()
                                current_area_id = 0

                                while len(seperate_from_current_area) != 0:
                                    start_zone, connection_tuple = seperate_from_current_area.popitem()

                                    # Seperate and connected has already been processed.
                                    #   Probably connected to area previously processed.
                                    if start_zone in visited_zones:
                                        continue

                                    if connection_tuple is not None:
                                        connection_zone, connection_area_id = connection_tuple

                                        # Connect both ways to make graph directed
                                        #   both ways
                                        add_to_set_value_to_dict(
                                            area_adjacency, connection_area_id, current_area_id
                                        )
                                        add_to_set_value_to_dict(
                                            area_adjacency, current_area_id, connection_area_id
                                        )

                                    area_queue = set()
                                    area_queue.add(start_zone)

                                    area = set()
                                    previous_good_zone = None

                                    while len(area_queue) != 0:
                                        current_zone = area_queue.pop()

                                        if current_zone in visited_zones:
                                            continue

                                        # zone_is_seperate = True
                                        # if current_zone.near_umbilic and start_zone.near_umbilic:
                                        #     zone_is_seperate = False
                                        #
                                        # elif current_zone.has_ridge and start_zone.has_ridge:
                                        #     # Min deviating or max deviating ridges are seperate zones
                                        #     if current_zone.min_deviating == start_zone.min_deviating \
                                        #             and current_zone.max_deviating == start_zone.max_deviating:
                                        #         zone_is_seperate = False

                                        zone_is_same = False
                                        if current_zone.near_umbilic == start_zone.near_umbilic \
                                                and current_zone.has_ridge == start_zone.has_ridge:
                                            zone_is_same = True

                                        if not zone_is_same:
                                            # Dont mark as visited yet as will be
                                            #   processed again later
                                            seperate_from_current_area[current_zone] = (
                                                previous_good_zone, current_area_id
                                            )
                                            continue

                                        visited_zones.add(current_zone)
                                        area.add(current_zone)
                                        previous_good_zone = current_zone

                                        for adjacent in final_depth_adjacency_graph[current_zone]:
                                            area_queue.add(adjacent)

                                    area_id_map[current_area_id] = (
                                        start_zone, area
                                    )
                                    if start_zone.has_ridge:
                                        ridge_areas.add(current_area_id)

                                    current_area_id += 1

                                # print(area_id_map.keys())
                                # print(area_adjacency)

                                print(len(ridge_areas))

                                # Find transitions of:
                                #   min ridge to umbilic to max ridge
                                # and
                                #   max ridge to umbilic to min ridge

                                for area_id, area in area_id_map.items():
                                    start_zone, zones = area
                                    if start_zone.has_ridge:
                                        print(area_id, len(zones))
                                        for zone in zones:
                                            zone.rasterize_to_image_buffer(
                                                image_buffer, image.size, area_id
                                            )
                            else:
                                # @TODO: If no special zones, what to do?
                                assert False

                        if False:
                            print("starting integration")

                            arc_length_distance = 0.04
                            step_size = 0.001

                            for zone in zones:
                                if zone.has_ridge:

                                    math_extensions.integrate_line_of_curvature_to_mesh_edges(
                                        output_bmesh,
                                        patches[patch_index],
                                        zone.get_center(),
                                        arc_length_distance,
                                        step_size,
                                        use_min_principle_direction=zone.min_deviating
                                    )
                                    math_extensions.integrate_line_of_curvature_to_mesh_edges(
                                        output_bmesh,
                                        patches[patch_index],
                                        zone.get_center(),
                                        arc_length_distance,
                                        step_size,
                                        use_min_principle_direction=zone.min_deviating,
                                        reverse=True
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

                                image_buffer[v_pixel][u_pixel] = (
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
                                        image_buffer[v_pixel][u_pixel] = (
                                            when_ratio_is_1_this_is_1, 0.0, 0.0, 1.0
                                        )
                                    else:
                                        image_buffer[v_pixel][u_pixel] = (
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

                                image_buffer[v_pixel][u_pixel] = (
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

                                found, value, parameters = math_extensions.umbilic_gradient_descent(
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

                                image_buffer[v_pixel_pos][u_pixel_pos] = (
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

                                found, value, parameters = math_extensions.umbilic_gradient_descent(
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
                                    image_buffer[v_pixel_pos][u_pixel_pos] = (
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

            if show_curvature_lines:
                view_layer = bpy.context.view_layer

                bpy.ops.object.mode_set(mode='OBJECT')

                old_active = view_layer.objects.active

                bpy.ops.object.select_all(action='DESELECT')

                view_layer.objects.active = output_object

                bpy.ops.object.mode_set(mode='EDIT')

                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.edge_face_add()
                bpy.ops.mesh.select_all(action='DESELECT')

                bpy.ops.object.mode_set(mode='OBJECT')

                view_layer.objects.active = old_active

                uv_layer = output_mesh.uv_layers.active
                if uv_layer is None:
                    uv_layer = output_mesh.uv_layers.new()

                uv_layer = uv_layer.data

                for poly in output_mesh.polygons:
                    for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                        vertex = output_mesh.vertices[output_mesh.loops[loop_index].vertex_index]
                        coordinate = vertex.co
                        uv_layer[loop_index].uv = (coordinate.x, coordinate.y)

            # for vertex in output_mesh.vertices:
            #     vertex.co = math_extensions.bezier_surface_at_parameters(
            #         patches[0],
            #         vertex.co.x,
            #         vertex.co.y
            #     )

        # Principle direction field
        if False:
            # Start new mesh from scratch
            output_bmesh = bmesh.new()

            output_object.vertex_groups.clear()
            # https://blender.stackexchange.com/a/117586
            vertex_group = output_object.vertex_groups.new(name="Max direction").index
            weight_layer = output_bmesh.verts.layers.deform.new("Max direction")

            # u_samples = v_samples = 40  # 200  # 50
            u_samples = v_samples = 10     # 200  # 50
            # u_bound = (0.05, 0.35)
            # v_bound = (0.05, 0.35)
            # u_bound = (0.2, 0.6)
            # v_bound = (0.5, 1.0)
            # u_bound = (0.1, 0.6)
            # v_bound = (0.2, 0.9)
            u_bound = (0.0, 1.0)
            v_bound = (0.0, 1.0)

            for patch in patches:
                for v_sample in range(v_samples):
                    for u_sample in range(u_samples):
                        u_base = u_sample / (u_samples - 1)
                        v_base = v_sample / (v_samples - 1)

                        u_val = math_extensions.lerp(u_bound[0], u_bound[1], u_base)
                        v_val = math_extensions.lerp(v_bound[0], v_bound[1], v_base)

                        min_principal_vector, max_principal_vector = \
                            math_extensions.min_max_principle_directions(
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

                        vector_scale = 0.0025 * 24  # * 3

                        normalize = True
                        if normalize:
                            min_principal_vector = min_principal_vector.normalized()
                            max_principal_vector = max_principal_vector.normalized()

                        min_principal_vector *= vector_scale
                        max_principal_vector *= vector_scale

                        parameters_vector = mathutils.Vector((u_val, v_val))
                        min_vector_end_parameters = parameters_vector + min_principal_vector
                        max_vector_end_parameters = parameters_vector + max_principal_vector

                        world_space = False
                        if world_space:
                            world_point = math_extensions.bezier_surface_at_parameters(
                                patch, u_val, v_val)

                            min_vector_end = math_extensions.bezier_surface_at_parameters(
                                patch, min_vector_end_parameters.x, min_vector_end_parameters.y
                            )

                            max_vector_end = math_extensions.bezier_surface_at_parameters(
                                patch, max_vector_end_parameters.x, max_vector_end_parameters.y
                            )
                        else:
                            world_point = mathutils.Vector((u_val, v_val, 0.0))
                            min_vector_end = min_vector_end_parameters.to_3d()
                            max_vector_end = max_vector_end_parameters.to_3d()

                        enable_min = True
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

        # Integrate lines of curvature
        if False:
            # Start new mesh from scratch
            output_bmesh = bmesh.new()

            color_layer = output_bmesh.edges.layers.color.new("Min/Max Color")

            u_samples = v_samples = 20
            u_bound = (0.05, 0.35)
            v_bound = (0.05, 0.35)

            for patch in patches:
                for v_sample in range(v_samples):
                    for u_sample in range(u_samples):
                        u_base = u_sample / (u_samples - 1)
                        v_base = v_sample / (v_samples - 1)

                        u_val = math_extensions.lerp(u_bound[0], u_bound[1], u_base)
                        v_val = math_extensions.lerp(v_bound[0], v_bound[1], v_base)

                        arc_length_distance = 1
                        step_size = 0.001
                        start_parameters = mathutils.Vector((u_val, v_val))

                        min_verts = math_extensions.integrate_line_of_curvature_to_mesh_edges(
                            output_bmesh,
                            patch,
                            start_parameters,
                            arc_length_distance,
                            step_size,
                            use_min_principle_direction=True
                        )
                        max_verts = math_extensions.integrate_line_of_curvature_to_mesh_edges(
                            output_bmesh,
                            patch,
                            start_parameters,
                            arc_length_distance,
                            step_size,
                            use_min_principle_direction=False,
                        )

                        for min_vert in min_verts:
                            min_vert[color_layer] = (0.188, 0.439, 0.945, 1.0)
                        for max_vert in max_verts:
                            max_vert[color_layer] = (0.878, 0.298, 0.298, 1.0)

            output_mesh = output_object.data
            output_bmesh.to_mesh(output_mesh)

    # Apply Bézier function
    if False:
        output_bmesh = bmesh.new()
        output_bmesh.from_mesh(output_object.data)

        for vertex in output_bmesh.verts:
            vertex.co = math_extensions.bezier_surface_at_parameters(
                patches[0],
                vertex.co.x,
                vertex.co.y
            )

        output_bmesh.to_mesh(output_object.data)

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


class HelloWorldOperator(bpy.types.Operator):
    bl_idname = "wm.hello_world"
    bl_label = "Minimal Operator"

    def execute(self, context):
        view_layer = context.view_layer
        active = view_layer.objects.active

        mesh = active.data

        # for vertex in mesh.vertices:
        #     vertex.co = math_extensions.bezier_surface_at_parameters(
        #         patches[0],
        #         vertex.co.x,
        #         vertex.co.y
        #     )

        return {'FINISHED'}


def register():
    """Call to register is made when addon is enabled."""
    bpy.utils.register_class(OBJECT_OT_add_bezier_surface)
    bpy.utils.register_class(HelloWorldOperator)

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
    bpy.utils.unregister_class(HelloWorldOperator)

    bpy.types.VIEW3D_MT_surface_add.remove(add_bezier_surface_button)

    bpy.app.handlers.depsgraph_update_pre.remove(cb_scene_update)


if __name__ == "__main__":
    register()
