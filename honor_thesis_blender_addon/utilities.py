import bmesh
import mathutils
import typing


def rasterize_point(image_buffer, point: mathutils.Vector, image_size, color):
    x = int(round(point.x * image_size[0]))
    y = int(round(point.y * image_size[1]))

    if x >= image_size[0] or x < 0:
        return

    if y >= image_size[1] or y < 0:
        return

    image_buffer[x][y] = color


def get_arbitrary_element_of_set(the_set):
    return next(iter(the_set))


def get_3x3_patches_from_sharp_border(mesh: bmesh.types.BMesh) -> list[set[bmesh.types.BMesh]]:
    """Find patch of 3x3 quads with 16 verticies_coordinates in control_bmesh defined by a sharp border."""
    output_patches = []
    unvisited_faces = set(mesh.faces)

    while len(unvisited_faces) != 0:
        # Start at random face
        start_face = unvisited_faces.pop()

        patch_faces = set()
        patch_corner_verticies = set()
        patch_corner_faces = set()
        patch_sharp_verticies = set()

        faces_queue = set()
        faces_queue.add(start_face)

        num_non_sharp_faces = 0

        # Depth first search (DFS)
        while len(faces_queue) != 0:
            current_face = faces_queue.pop()

            # If traversal for edge boundary hit non-quad, then this is not a
            #   3x3 patch
            if len(current_face.verts) != 4:
                break

            patch_faces.add(current_face)

            # If traversal has hit more than 9 quads, then this not a 3x3 patch
            if len(patch_faces) > 9:
                break

            temp_patch_corner_verticies = None

            non_sharp_edges = []
            sharp_edges = []
            for edge in current_face.edges:
                if edge.smooth:
                    # Add to queue non-sharp edges of face
                    non_sharp_edges.append(edge)
                else:
                    sharp_edges.append(edge)

                    sharp_verticies = set(vertex for vertex in edge.verts)
                    patch_sharp_verticies = patch_sharp_verticies.union(sharp_verticies)

                    # Get verticies_coordinates shared between sharp edges in face
                    if temp_patch_corner_verticies is None:
                        temp_patch_corner_verticies = sharp_verticies
                    else:
                        temp_patch_corner_verticies = temp_patch_corner_verticies.intersection(
                            sharp_verticies)

            if len(non_sharp_edges) == 4:
                num_non_sharp_faces += 1

            if len(non_sharp_edges) == 2:
                # Should only be 1 corner vertex in corners of 3x3 patch
                if temp_patch_corner_verticies is not None and len(temp_patch_corner_verticies) > 0:
                    patch_corner_verticies.add(temp_patch_corner_verticies.pop())

                patch_corner_faces.add(current_face)

            # Acquire linked faces_vertex_indicies to traverse next
            for non_sharp_edge in non_sharp_edges:
                for link_face in non_sharp_edge.link_faces:
                    # Do not visit already visited faces_vertex_indicies
                    if link_face not in patch_faces:
                        faces_queue.add(link_face)

        if len(patch_faces) == 9 and num_non_sharp_faces == 1:
            # Condition for valid patch boundary
            output_patches.append(patch_corner_faces)

        unvisited_faces = unvisited_faces.difference(patch_faces)

    return output_patches


def edge_is_sharp(edge):
    return not edge.smooth


def get_3x3_patch_loops_in_topological_order(
    patch_sharp_corner_faces: set[bmesh.types.BMFace]
) -> list[list[bmesh.types.BMLoop]]:
    output_loops = [[None] * 4 for _ in range(4)]

    # Secondary edge is horizontal, start of vertical traversal
    primary_sharp = None
    secondary_sharp = None

    # Choose starting corner based on order of sharp edges around the
    #   loop
    # Loops seem to go counter-clockwise
    for corner_face in patch_sharp_corner_faces:
        primary_sharp = None
        secondary_sharp = None
        good = True

        loop_index = 0
        for loop in corner_face.loops:
            good = False

            if loop_index == 0 and edge_is_sharp(loop.edge):
                secondary_sharp = loop
                good = True
            elif loop_index == 1 and edge_is_sharp(loop.edge):
                primary_sharp = loop
                good = True
            elif loop_index == 2 and not edge_is_sharp(loop.edge):
                good = True
            elif loop_index == 3 and not edge_is_sharp(loop.edge):
                good = True

            if not good:
                break

            loop_index += 1

        # Found a good corner
        if good:
            break

    # Build 2D table of verticies_coordinates by doing a edge ring traversal
    #   for every edge in the orthogonal
    #       (wrong word as direction between edges can be skewed)
    #   edge ring traversal
    start_loop = secondary_sharp
    for secondary_edge_num in range(0, 3):
        start_loop = start_loop.link_loop_next

        # print("aaa")
        # print(start_loop.edge.index)
        # print("begin")

        loop = start_loop

        # print(loop.vert.index)

        for edge_num in range(0, 3):
            for loop_num in range(0, 2):
                # [y][x]
                y = edge_num
                x = secondary_edge_num + loop_num
                output_loops[y][x] = loop

                loop = loop.link_loop_next

            if edge_num != 2:
                loop = next(iter(loop.link_loops))
            else:
                output_loops[3][secondary_edge_num + 1] = loop
                loop = loop.link_loop_next

        output_loops[3][secondary_edge_num] = loop

        start_loop = start_loop.link_loop_next

        if secondary_edge_num != 2:
            start_loop = next(iter(start_loop.link_loops))

    return output_loops


def grid_mesh(
        generator_function: typing.Callable[[float, float], mathutils.Vector],
        u_num_output_verticies: int,
        v_num_output_verticies: int
):
    verticies_coordinates = []

    for y_control_point in range(v_num_output_verticies):
        # Iterate over each u row
        for x_control_point in range(u_num_output_verticies):
            u = x_control_point / (u_num_output_verticies - 1)
            v = y_control_point / (v_num_output_verticies - 1)

            vertex = generator_function(u, v)
            verticies_coordinates.append(vertex)

    faces_vertex_indicies = []

    num_faces_u = u_num_output_verticies - 1
    num_faces_v = v_num_output_verticies - 1

    for face_v in range(num_faces_v):
        for face_u in range(num_faces_u):
            current_row_offset = face_v * u_num_output_verticies
            next_row_offset = current_row_offset + u_num_output_verticies
            current_x_pos = face_u

            face_vertex_indicies = [
                current_row_offset + current_x_pos,
                current_row_offset + current_x_pos + 1,
                next_row_offset + current_x_pos + 1,
                next_row_offset + current_x_pos
            ]

            faces_vertex_indicies.append(face_vertex_indicies)

    return (verticies_coordinates, faces_vertex_indicies)


def add_grid_mesh_to_bmesh(
    bmesh_to_modify: bmesh.types.BMesh,
    generator_function: typing.Callable[[float, float], mathutils.Vector],
    u_num_output_verticies: int,
    v_num_output_verticies: int
):
    verticies_points, faces_vertex_indicies = grid_mesh(
        generator_function,
        u_num_output_verticies,
        v_num_output_verticies
    )

    bmesh_verticies = [bmesh_to_modify.verts.new(vertex_point) for vertex_point in verticies_points]
    bmesh_to_modify.verts.index_update()

    bmesh_faces = []
    for face_vertex_indicies in faces_vertex_indicies:
        face_vertex_references = [
            bmesh_verticies[index] for index in face_vertex_indicies
        ]
        bmesh_faces.append(bmesh_to_modify.faces.new(face_vertex_references))

    return (bmesh_verticies, bmesh_faces)
