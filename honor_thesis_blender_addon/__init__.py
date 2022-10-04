"""2022 Honors Thesis Blender Addon."""

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector

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

    # Generate subdivided axis-aligned plane
    u_num_control_points = 4
    v_num_control_points = 4

    verticies = []

    for y_control_point in range(v_num_control_points):
        # Iterate over each u row
        for x_control_point in range(u_num_control_points):
            u = x_control_point / (u_num_control_points - 1)
            v = y_control_point / (v_num_control_points - 1)

            x_pos = (u * 2) - 1
            y_pos = (v * 2) - 1

            vertex = Vector((x_pos * scale_x, y_pos * scale_y, 0))
            verticies.append(vertex)


    edges = []

    faces = []

    num_faces_u = u_num_control_points - 1
    num_faces_v = v_num_control_points - 1

    for face_v in range(num_faces_v):
        for face_u in range(num_faces_u):
            current_row_offset = face_v * u_num_control_points
            next_row_offset = current_row_offset + u_num_control_points
            current_x_pos = face_u

            loop_cycle = [
                current_row_offset + current_x_pos,
                current_row_offset + current_x_pos + 1,
                next_row_offset + current_x_pos + 1,
                next_row_offset + current_x_pos
            ]

            faces.append(loop_cycle)

    control_mesh = bpy.data.meshes.new(name="Bézier Control Mesh")

    # output_mesh will not be populated with anything (blank mesh)
    output_mesh = bpy.data.meshes.new(name="Bézier Output Mesh")

    # from_pydata:
    #   https://docs.blender.org/api/current/bpy.types.Mesh.html#bpy.types.Mesh.from_pydata
    control_mesh.from_pydata(verticies, edges, faces)

    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    # object_data_add:
    #   https://docs.blender.org/api/current/bpy_extras.object_utils.html#bpy_extras.object_utils.object_data_add
    created_control_grid = object_data_add(context, control_mesh, operator=self, name="Bézier Surface")
    created_output_object = object_data_add(context, output_mesh, operator=self, name="Bézier Output Object")

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

def create_and_replace_output_mesh(control_mesh: bpy.types.Mesh, output_mesh_to_replace: bpy.types.Mesh):
    control_bmesh = bmesh.from_edit_mesh(control_mesh)

    print(control_bmesh)

    # Find patch of 3x3 quads with 16 verticies in control_bmesh defined by a sharp border

    unvisited_faces = set(control_bmesh.faces)

    patches = []

    while len(unvisited_faces) != 0:
        # Start at random face
        start_face = unvisited_faces.pop()

        patch = []

        visited_faces = set()
        faces_queue = set()
        faces_queue.add(start_face)

        num_non_sharp_faces = 0
        set_of_corner_edges = None

        while len(faces_queue) != 0:
            current_face = faces_queue.pop()

            # Traversal for edge boundary hit non-quad
            if len(current_face.verts) != 4:
                break

            visited_faces.add(current_face)

            # If traversal has hit more than 9 quads, then this not a 3x3 patch
            if len(visited_faces) > 9:
                break

            # Find non-sharp edges of face
            non_sharp_edges = []
            for edge in current_face.edges:
                if edge.smooth:
                    non_sharp_edges.append(edge)

            if len(non_sharp_edges) == 4:
                num_non_sharp_faces += 1

            if len(non_sharp_edges) == 2 and set_of_corner_edges is None:
                set_of_corner_edges = (current_face, non_sharp_edges)

            # Acquire linked faces to traverse next
            for non_sharp_edge in non_sharp_edges:
                for link_face in non_sharp_edge.link_faces:
                    # Do not visit already visited faces
                    if link_face not in visited_faces:
                        faces_queue.add(link_face)

        if len(visited_faces) == 9 and num_non_sharp_faces == 1:
            # Valid patch boundary
            print("valid")

            # For quads, Traversal of 2 loops acquires the edge opposite to
            #   start loop regardless of direction of loop.

            # 1. Start at one edge of a corner, traverse edge ring while
            #   storing each sharp edge.  This will define the the "u" border
            #   verticies.
            # 2. Start at the other edge of the corner, traverse edge ring
            #   while storing each sharp edge.  This will define the the "u"
            #   border verticies.

        unvisited_faces = unvisited_faces.difference(visited_faces)

    # Create new output mesh
    output_bmesh = bmesh.new()
    output_bmesh.from_mesh(output_mesh_to_replace)

    # Replace output mesh with newly generated one
    output_bmesh.to_mesh(output_mesh_to_replace)
    output_bmesh.free()

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
            output_mesh = output_object.data

            create_and_replace_output_mesh(control_mesh, output_mesh)

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
