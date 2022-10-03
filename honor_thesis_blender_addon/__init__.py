"""2022 Honors Thesis Blender Addon."""

import bpy
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

    verts = [
        Vector((-1 * scale_x, 1 * scale_y, 0)),
        Vector((1 * scale_x, 1 * scale_y, 0)),
        Vector((1 * scale_x, -1 * scale_y, 0)),
        Vector((-1 * scale_x, -1 * scale_y, 0)),
    ]

    edges = []
    faces = [[0, 1, 2, 3]]

    control_mesh = bpy.data.meshes.new(name="Bézier Control Mesh")
    output_mesh = bpy.data.meshes.new(name="Bézier Output Mesh")

    # from_pydata:
    #   https://docs.blender.org/api/current/bpy.types.Mesh.html#bpy.types.Mesh.from_pydata
    control_mesh.from_pydata(verts, edges, faces)

    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    # object_data_add:
    #   https://docs.blender.org/api/current/bpy_extras.object_utils.html#bpy_extras.object_utils.object_data_add
    created_control_grid = object_data_add(context, control_mesh, operator=self, name="Bézier Surface")
    created_output_object = object_data_add(context, output_mesh, operator=self, name="Bézier Output Mesh")

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

def add_bezier_surface_button(self, context):
    self.layout.operator(
        OBJECT_OT_add_bezier_surface.bl_idname,
        text="Bézier Surface",
        icon='SURFACE_NSURFACE')


def register():
    """Call to register is made when addon is enabled."""
    bpy.utils.register_class(OBJECT_OT_add_bezier_surface)

    # "Extending Menus" Blender API
    #       https://docs.blender.org/api/current/bpy.types.Menu.html#extending-menus
    # "VIEW3D_MT_surface_add determined by Python Blender UI
    #       Source: https://github.com/blender/blender/blob/482d431bb6735e8206961bd1115d2be7e63572b1/release/scripts/startup/bl_ui/space_view3d.py#L2122
    bpy.types.VIEW3D_MT_surface_add.append(add_bezier_surface_button)


def unregister():
    """Call to unregister is made when addon is disabled."""
    bpy.utils.unregister_class(OBJECT_OT_add_bezier_surface)
    bpy.types.VIEW3D_MT_surface_add.remove(add_bezier_surface_button)


if __name__ == "__main__":
    register()
