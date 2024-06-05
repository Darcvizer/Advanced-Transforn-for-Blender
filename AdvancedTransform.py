import math
import blf
import bmesh
import bpy
import gpu
import mathutils
import rna_keymap_ui
from bpy import context
from bpy.props import EnumProperty, BoolProperty
from bpy.utils import register_class, unregister_class
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix
from mathutils import Vector as V
from mathutils.geometry import intersect_line_plane

bl_info = {
    "name": "Advanced Transform 2",
    "location": "View3D > Advanced Transform 2",
    "description": "Advanced Transform 2",
    "author": "Vladislav Kindushov(Darcvizer)",
    "version": (0, 5, 0),
    "blender": (4, 1, 0),
    "category": "View3D"}

gv0 = V((0,0,0))
"""Vector: V((0,0,0))"""
gvx = V((1,0,0))
"""Vector: V((1,0,0))"""
gvy = V((0,1,0))
"""Vector: V((0,1,0))"""
gvz = V((0,0,1))
"""Vector: V((0,0,1))"""

DubegMode = True
CirclePointDivider = 3
RingWidth = 0.25
RingRadius = 1
AxisColor = lambda axis, alpha=0.3 : (0.8, 0.2, 0.2, alpha) if axis == 0 else ((0.1, 0.6, 0.1, alpha) if axis == 1 else ((0.0, 0.3, 0.6, alpha) if axis == 2 else (1, 1, 1, alpha)))
Is_EditMesh = lambda: bpy.context.mode == 'EDIT_MESH'
Is_3d = lambda: bpy.context.space_data.type == "VIEW_3D"
Is_3d_var = None
"""Need in cases like filling header whne current context isn't 3d"""

def is_3d_required(func):
    def wrapper(self, *args, **kwargs):
        if Is_3d():
            return func(self, *args, **kwargs)
        else:
            parent_class = super(self.__class__, self)
            parent_method = getattr(parent_class, func.__name__)
            return parent_method(*args, **kwargs)
    return wrapper

def is_edit_mesh(func):
    def wrapper(self, *args, **kwargs):
        if Is_EditMesh():
            return func(self, *args, **kwargs)
        else:
            parent_class = super(self.__class__, self)
            parent_method = getattr(parent_class, func.__name__)
            return parent_method(*args, **kwargs)
    return wrapper

def GetCursorPosition(region = False):
    if Is_3d():
        return bpy.context.scene.cursor.location.to_3d().copy()
    else: 
        if region:
            vector = bpy.context.space_data.cursor_location.copy()
            pivot = V(bpy.context.region.view2d.view_to_region(vector.x,vector.y))
            return pivot.to_3d()
        else: 
            return bpy.context.space_data.cursor_location.copy().to_3d()

def SetCursorPosition(value, region = False):
    if Is_3d(): bpy.context.scene.cursor.location = value.to_3d().copy()
    else: 
        if region:
            value = V(bpy.context.region.view2d.region_to_view(value.x, value.y))
        bpy.context.space_data.cursor_location = value.to_2d().copy()
def SetCursorToSelection():
    if Is_3d(): bpy.ops.view3d.snap_cursor_to_selected()
    else: bpy.ops.uv.snap_cursor(target='SELECTED')

def deb(value,dis=None, pos=None, dir=None, forced=False):
    """Debug function
    text: Value for print | It is general prtin()
    dis: sting = None | Discription for print
    pos: Vector = None |Position for created object
    dir: Vector = None Should be normalized!!!| Direction for created object
    forced: forced=False | Forced create object
    """
    def CreateObjects():
        if Is_EditMesh():
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.mesh.primitive_plane_add()
        plane = bpy.context.object
        #bpy.context.collection.objects.link(plane) 
        bpy.ops.object.empty_add()
        null = bpy.context.object
        #bpy.context.collection.objects.link(null)
        bpy.context.object.empty_display_type = 'SINGLE_ARROW'
        bpy.context.object.empty_display_size = 3   
        plane.location = pos if pos != None else gv0
        plane.matrix_world = dir.to_track_quat('Z', 'Y').to_matrix().to_4x4() if dir != None else gvz.to_track_quat('Z', 'Y').to_matrix().to_4x4()
        plane.name = "ADHelpPlane" 
        null.name = "ADHelpNull"
        bpy.ops.object.constraint_add(type='COPY_TRANSFORMS')
        bpy.context.object.constraints["Copy Transforms"].target = plane    
        #bpy.ops.object.mode_set(mode='EDIT_MESH')

    if DubegMode:
        print(dis if dis != None and isinstance(dis, str) else "",': ' , value)
        plane = bpy.data.objects.get("ADHelpPlane")
        null = bpy.data.objects.get("ADHelpNull")
        if  pos != None or dir != None:
            if (plane or null):
                if null and dir != None:
                    null.matrix_world = dir.to_track_quat('Z', 'Y').to_matrix().to_4x4()
                if null and pos != None:
                    null.location = pos
            else:
                CreateObjects()
        if forced:
            CreateObjects()
    return 0

def GetBestAxisInMatrix(matrix, axis_vector):
    """Find max similar axis in matrix"""
    index_best_axis = [abs(axis_vector.dot(matrix.col[i])) for i in range(0,3)] # Get dot for all axes
    index_best_axis_index = index_best_axis.index(max(index_best_axis))
    return index_best_axis_index

def GetViewDirection():
    if Is_3d():
        return bpy.context.region_data.view_rotation @ V((0, 0, -1))
    else: 
        return gvz

def GetNormalForIntersectionPlane(TransfromOrientationMatrix):
    """Getting the closest axis to camera direction"""
    view_direction = GetViewDirection()
    index_best_axis_index = GetBestAxisInMatrix(TransfromOrientationMatrix, view_direction)
    normal_intersect_plane = (TransfromOrientationMatrix.col[index_best_axis_index]).normalized()# do invert for normal because we need derection to camera
    return normal_intersect_plane.to_3d()

def GetPivotPointPoistion():
    # best wat to get point use cursor, because using loop if user selected 1000 meshes or make bmesh too slowly    
    condition = lambda name: bpy.context.scene.tool_settings.transform_pivot_point == name if Is_3d() else bpy.context.space_data.pivot_point == name

    original_cursore_position = GetCursorPosition()

    bpy.ops.view3d.snap_cursor_to_active() if condition('ACTIVE_ELEMENT') else SetCursorToSelection()
    new_pivot_point = GetCursorPosition(region=True)
    SetCursorPosition(original_cursore_position)
    return GetCursorPosition(region=True) if condition('CURSOR') else new_pivot_point

def MakeCustomTransformOrientation(use_view=False):
    try: # Blender can't make custom transform orientation if selected all elements 
        matrix_transform_orientation = Matrix()
        temp = bpy.context.scene.transform_orientation_slots[0].type
        bpy.ops.transform.create_orientation(name="AdvancedTransform", use_view=use_view, use=True,overwrite=True)
        if bpy.context.scene.transform_orientation_slots[0].type == "AdvancedTransform":
            matrix_transform_orientation = bpy.context.scene.transform_orientation_slots[0].custom_orientation.matrix.copy()
            bpy.ops.transform.delete_orientation()
            bpy.context.scene.transform_orientation_slots[0].type = temp
            return matrix_transform_orientation
    except: 
        return matrix_transform_orientation

def GetTransfromOrientationMatrix():
    """Because there is no way to get the current transformation matrix, I have to use this code"""

    condition = lambda value: bpy.context.scene.transform_orientation_slots[0].type == value

    if condition('GLOBAL') or condition('GIMBAL'):
        matrix_transform_orientation = Matrix()
    elif condition('LOCAL'):
        matrix_transform_orientation = bpy.context.active_object.rotation_euler.to_matrix().copy()
    elif condition('NORMAL'):
        if bpy.context.mode == 'OBJECT':
            matrix_transform_orientation = bpy.context.active_object.rotation_euler.to_matrix().copy()
        elif bpy.context.mode == 'EDIT_MESH':
            matrix_transform_orientation = MakeCustomTransformOrientation()
    elif condition('VIEW'):
        matrix_transform_orientation = MakeCustomTransformOrientation(use_view=True)
    elif condition('CURSOR'):
        matrix_transform_orientation = bpy.context.scene.cursor.matrix.copy()
    else:
        matrix_transform_orientation = bpy.context.scene.transform_orientation_slots[0].custom_orientation.matrix.copy()
    return matrix_transform_orientation.copy()

def GetMouseLocation(pivot_point, normal, matrix,  event):
    """convert mouse pos to 3d point over plane defined by origin and normal"""
    region = bpy.context.region
    rv3d = bpy.context.region_data
    coord = event.mouse_region_x, event.mouse_region_y

    view_vector_mouse = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin_mouse = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    view_vector_mouse = ray_origin_mouse + view_vector_mouse

    loc = intersect_line_plane(ray_origin_mouse, view_vector_mouse, pivot_point, normal, False)
    return loc

def GetIndexOfMaxValueInVector(vector):
    """Get axis by mouse direaction\n
    normal: vector\n
    return_index return name axis (x,y,z) or index (0,1,2)
    """
    index_of_max_value = max(range(len(vector)), key=lambda i: abs(vector[i])) # get index of maximal value at vector
    return index_of_max_value

def GetMouseDirectionAxis(pivot, point_from, point_to, matrix, force_multi_axis = False):
    use_multi_axis = bool(get_addon_preferences().MultiAxisDrag)
    if use_multi_axis or force_multi_axis:
        return GetMouseDirectionAxis3D(pivot, point_from, point_to, matrix)
    else:
        return GetMouseDirectionAxisByPlane(point_from, point_to, matrix)

def GetMouseDirectionAxisByPlane(point_from, point_to, matrix):
    """Get mouse direction by 2 saved points, also if we have specific transform orientation we have to rotate vector by orientation matrix
    point_from: Vector | mouse point 1
    point_to: Vector | mouse point 2
    matrix: Matrix | Matrix for rotation vector
    """
    direction = (matrix.to_3x3().inverted() @ (point_from - point_to).normalized()) # Rotate move direction to transform orientation matrix
    return GetIndexOfMaxValueInVector(direction)

def GetMouseDirectionAxis3D(pivot, point_from, point_to, matrix):
    context = bpy.context
    region = context.region
    rv3d = context.region_data
    # Get 2d mouse data
    if Is_3d():
        mouse_start = view3d_utils.location_3d_to_region_2d(region,rv3d, point_from)
        mouse_end = view3d_utils.location_3d_to_region_2d(region,rv3d, point_to)
        mouse_dir = (mouse_start - mouse_end).normalized()

        pivot2d = mouse_end = view3d_utils.location_3d_to_region_2d(region,rv3d, pivot)

        # Get 2d matrix directions
        x = (matrix.col[0].to_3d() * 2 + pivot)
        y = (matrix.col[1].to_3d() * 2 + pivot)
        z = (matrix.col[2].to_3d() * 2 + pivot)
        x = view3d_utils.location_3d_to_region_2d(region,rv3d, x) 
        x = (x - pivot2d).normalized()
        y = view3d_utils.location_3d_to_region_2d(region,rv3d, y)
        y = (y - pivot2d).normalized()
        z = view3d_utils.location_3d_to_region_2d(region,rv3d, z)
        z = (z - pivot2d).normalized()
        axes = [x,y,z]
    else:
        mouse_dir = (point_from - point_to).normalized()
        axes = [gvx,gvy]
    index_best_axis = [abs(mouse_dir.dot(i)) for i in axes]
    index_best_axis_index = index_best_axis.index(max(index_best_axis))
    return index_best_axis_index

def GetPivotPointPoistionUV():
    condition = lambda name: bpy.context.space_data.pivot_point == name
    original_cursore_position = GetCursorPosition()
    if not condition('CURSOR') :
        bpy.ops.uv.snap_cursor(target='SELECTED')
        pivot = GetCursorPosition()
        SetCursorPosition(original_cursore_position)
        pivot = V(context.region.view2d.view_to_region(pivot.x, pivot.y))
        return pivot
    else:
        pivot = original_cursore_position
        pivot = V(context.region.view2d.view_to_region(pivot.x, pivot.y))
        return pivot
    
def GetMouseDirectionUV(poin1, point2):
    direction = (point2 - poin1).normalized()
    return GetIndexOfMaxValueInVector(direction)

def SpawnCursorByRaycast(mouse_position, set_poisition = False, set_orientation = False, free_meshes=False):
    """Use raycast find the closest element on face
    Return position and rotation(euler)"""
    if free_meshes: SpawnCursorByRaycast.objects = []; return None # it is optimization
    if not hasattr(SpawnCursorByRaycast, 'objects'):
        SpawnCursorByRaycast.objects = []
    SpawnCursorByRaycast
    def find_closest_vector(values, target_vector):
        closest_vector = None
        closest_key = None
        closest_index = 0
        min_distance = float('inf')

        for key, vectors in values.items():
            for vector in vectors:
                distance = (target_vector - vector).length
                if distance < min_distance:
                    min_distance = distance
                    closest_vector = vector
                    closest_key = key
                    closest_index = vectors.index(vector)

        return closest_key, closest_vector, closest_index
    
    def create_orientation_matrix(normal, right=None):
        normal = normal.normalized() 
        if right is None:
            arbitrary_vector = gvx if abs(normal.x) < 0.9 else gvy
            x_axis = normal.cross(arbitrary_vector).normalized()
            y_axis = normal.cross(x_axis).normalized()
            return Matrix((x_axis, y_axis, normal)).transposed()
        
        else:
            x_axis = right.cross(normal).normalized()
            return Matrix((x_axis, right, normal)).transposed()
    
    context = bpy.context
    depsgraph = context.evaluated_depsgraph_get()
    region = context.region
    rv3d = context.region_data
    context.region_data.view_location
    camera_location = context.region_data.view_matrix.inverted().to_translation().to_3d()
    deb(camera_location,"camera_location")
    mouse_direction = (mouse_position - camera_location).normalized()

    mouse_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_position)
    RESULT, LOCATION, NORMAL, INDEX, OBJECT, MATRIX = context.scene.ray_cast(depsgraph, origin=camera_location, direction=mouse_direction)

    points = {"vertices":[], "edges":[], "face":[]}
    normals = {"vertices":[], "edges":[], "face":[]}
    
    if RESULT:
        if Is_EditMesh():
            if not OBJECT in SpawnCursorByRaycast.objects:
                OBJECT.update_from_editmode()
                SpawnCursorByRaycast.objects.append(OBJECT)
        
        #get_length = lambda e, lengthest_edge: e if (OBJECT.data.vertices[e[0]].co - OBJECT.data.vertices[e[1]].co).length > (OBJECT.data.vertices[lengthest_edge[0]].co - OBJECT.data.vertices[lengthest_edge[1]].co).length else lengthest_edge
        lengthest_edge = [gv0,gv0]
        vectors = []
        for e in OBJECT.data.polygons[INDEX].edge_keys:
                v1 = OBJECT.data.vertices[e[0]]
                v2 = OBJECT.data.vertices[e[1]]
                # Append element directions for transform orientation, Translate wrld normal to local because function gizmo use local coordinates    
                normals["vertices"].append(MATRIX.to_3x3() @ v1.normal)
                normals["vertices"].append(MATRIX.to_3x3() @ v2.normal)
                normals["edges"].append(((MATRIX @ v1.co) - (MATRIX @ v2.co)).normalized())

                # Append element position for pivot point

                v1 = MATRIX @ v1.co
                v2 = MATRIX @ v2.co
                edge = (v1 + v2)/2
                points["vertices"].append(v1)
                points["vertices"].append(v2)
                points["edges"].append(edge)
                vectors.append(v1)
                vectors.append(v2)
                vectors.append(edge)
                lengthest_edge = [v1,v2] if (v1 - v2).length > (lengthest_edge[0] - lengthest_edge[1]).length else lengthest_edge

        # append face normal
        normals["face"].append(NORMAL)
        # append center of face
        center = sum(vectors, mathutils.Vector()) / len(vectors)
        points["face"].append(center)
        # get index the closest element to mouse
        closest_key, closest_pivot, closest_index = find_closest_vector(points, LOCATION)
        closest_normal = normals[closest_key][closest_index]
        matrix = Matrix()
        if set_orientation:
            if closest_key == "face":
                y = (lengthest_edge[1] - lengthest_edge[0]).normalized()
                matrix = create_orientation_matrix(closest_normal, y).to_4x4()
            elif closest_key == "edges":
                deb("edges",closest_normal)
                matrix = create_orientation_matrix(closest_normal, normals["face"][0]).to_4x4()
            else:
                matrix = create_orientation_matrix(closest_normal).to_4x4()
            bpy.context.scene.cursor.matrix = matrix
        if set_poisition:
            SetCursorPosition(closest_pivot)


        return closest_pivot, matrix
    else: return None , None

def QuickMoveToMouse(mouse_position):
    temp_cursor_pos = GetCursorPosition()
    SetCursorPosition(mouse_position)
    bpy.ops.view3d.snap_selected_to_cursor(use_offset=True)
    SetCursorPosition(temp_cursor_pos)

#------------------Utulity------------------------#

class Headers():
    def MoveHeader(self, context):
        layout = self.layout
        if Is_3d_var:
            layout.label(text="Axis Constrain", icon="MOUSE_LMB")
            layout.label(text="Plane Constrain", icon="MOUSE_RMB")
            layout.label(text="Free Move", icon="MOUSE_MMB")
            layout.label(text="Temporary Pivot", icon="EVENT_SHIFT")
            layout.label(text="Temporary Orientation", icon="EVENT_ALT")
            layout.label(text="Free Move With Snapping to Surface", icon="EVENT_SPACEKEY")
        else:
            deb("Is 2d")
            layout.label(text="Axis Constrain", icon="MOUSE_LMB")
            layout.label(text="Plane Move", icon="MOUSE_RMB")
            layout.label(text="Temporary Pivot", icon="EVENT_SHIFT")

    def ScaleHeader(self, context):
        layout = self.layout
        if Is_3d_var:
            layout.label(text="Axis Constrain", icon="MOUSE_LMB")
            layout.label(text="Plane Constrain", icon="MOUSE_RMB")
            layout.label(text="Free Scale", icon="MOUSE_MMB")
            layout.label(text="Temporary Pivot", icon="EVENT_SHIFT")
            layout.label(text="Temporary Orientation", icon="EVENT_ALT")
            layout.label(text="Mirror By Axis", icon="EVENT_CTRL")
            layout.label(text="Scale To 0 By Axis", icon="EVENT_SPACEKEY")
        else:
            layout.label(text="Axis Constrain", icon="MOUSE_LMB")
            layout.label(text="Free Scale", icon="MOUSE_RMB")
            layout.label(text="Temporary Pivot", icon="EVENT_SHIFT")
            layout.label(text="Mirror By Axis", icon="EVENT_CTRL")
            layout.label(text="Scale To 0 By Axis", icon="EVENT_SPACEKEY")

    def RotationHeader(self, context):
        layout = self.layout
        if Is_3d_var:
            layout.label(text="Step Rotation", icon="MOUSE_LMB")
            layout.label(text="Axis Rotation", icon="MOUSE_RMB")
            layout.label(text="View Rotation", icon="MOUSE_MMB")
            layout.label(text="Temporary Pivot", icon="EVENT_SHIFT")
            layout.label(text="Temporary Orientation", icon="EVENT_ALT")
            layout.label(text="Constrain Rotation", icon="EVENT_CTRL")
            layout.label(text="Trackball", icon="EVENT_SPACEKEY")
        else:
            layout.label(text="Step Rotation", icon="MOUSE_LMB")
            layout.label(text="Axis Rotation", icon="MOUSE_RMB")
            layout.label(text="Temporary Pivot", icon="EVENT_SHIFT")

    def RotationStepOff(self, context):
        layout = self.layout
        step_value = str(int(get_addon_preferences().Snapping_Step))
        layout.label(text="Step Rotation " + step_value + " On", icon="EVENT_SHIFT")
    def RotationStepOn(self, context):
        layout = self.layout
        layout.label(text="Step Rotation Off", icon="EVENT_SHIFT")


class ShaderUtility():
    def __init__(self, matrix: Matrix, pivot: V, axis: str):
        self.UpdateData(matrix, pivot, axis)

        self.GetForwardVector = lambda: V((0, -1, 0))
        self.ApplyScale = lambda scale, arr: [(i @ mathutils.Matrix.Scale(scale, 4)) for i in arr]
        self.ApplyRotationWithOffset = lambda angle, arr, axis: [(i - self.Pivot) @ mathutils.Matrix.Rotation(angle, 3, axis) + self.Pivot for i in arr]
        self.ApplyRotation = lambda angle, arr, axis: [(i) @ mathutils.Matrix.Rotation(angle, 3, axis) for i in arr]
        self.ApplyOffsetByNormal = lambda value, arr:  [value * (i - gv0).normalized() for i in arr]
        self.SignedAngle= lambda v1, v2, axis: v1.angle(v2) * (1 if axis.dot(v1.cross(v2)) >= 0 else -1)
        pass

    def UpdateData(self, matrix, pivot, axis):
        self.AxisInMatrix = axis
        self.Matrix = matrix
        self.Pivot = pivot.to_3d().copy()
        self.DirectionVector = matrix.col[self.AxisInMatrix].to_3d() 

    def RTM(self, vertices, override_dir_vec = None, override_cur_dir_vect = None, flip_current_forward = False):
        """Rotate Vector By Transform Orientation Matrix"""
        # Firstly rotate from Y to desired at world axis
        matrix =  Matrix().to_3x3()

        desired_direction = matrix.col[self.AxisInMatrix] if override_dir_vec == None else override_dir_vec
        current_dir_vec = self.GetForwardVector() if override_cur_dir_vect == None else override_cur_dir_vect# i don't know why but axies X and Z inverted and i must flip them
        if flip_current_forward: current_dir_vec *= -1

        angle = current_dir_vec.angle(desired_direction) # get angle between original direction and desired rirection

        axis_for_rotation = current_dir_vec.cross(desired_direction) # Axis for rotation
        if axis_for_rotation.length == 0:
            axis_for_rotation = matrix.col[self.AxisInMatrix]


        vertices = self.ApplyRotation(angle, vertices, axis_for_rotation)
        # Transfer to local axis in matrix
        matrix = self.Matrix.inverted()

        # Crutch for the Y axis, I don’t know that it breaks
        if self.AxisInMatrix == 1:
            matrix = self.Matrix.inverted() * -1 if flip_current_forward else self.Matrix.inverted()

        vertices = [i.to_3d() @ matrix + self.Pivot  for i in vertices]
        return vertices

    def ViewSize(self, vertices):
        """Calculate screen size"""
        if Is_3d():

            region = bpy.context.region
            rv3d = bpy.context.region_data

            z = bpy.context.region_data.view_rotation @ gvy
            y = bpy.context.region_data.view_rotation @ gvx

            z_point = z * 1 + self.Pivot 
            y_point = y * 1 + self.Pivot 

            z_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, z_point)
            y_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, y_point)

            length = (z_2d - y_2d).length
            scale_factor = 110 / length

            scale_matrix = mathutils.Matrix.Scale(scale_factor,3)

            return [((v-self.Pivot)@ scale_matrix)+self.Pivot for v in vertices]

            view_distance = bpy.context.region_data.view_distance

            # old varios with distance to view point
            lerp = lambda a, b, t: (1 - t) * a + t * b # Thanks laundmo for example https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56
            scale_factor = lerp(0 ,1,view_distance / 10)

            scale_matrix = mathutils.Matrix.Scale(scale_factor,3)
            return [((v-self.Pivot)@ scale_matrix)+self.Pivot for v in vertices]
        else: return vertices
    
    def Facing(self, vertices):
        # Get view camera location
        if Is_3d():
            camera_location = bpy.context.region_data.view_matrix.inverted().to_translation()

            # Get Up Direction of shape
            v1=vertices[0]
            v2=vertices[1]
            v3=vertices[2]

            vec1 = v2 - v1
            vec2 = v3 - v1

            up =vec1.cross(vec2).normalized()

            # Get camera direction
            direction_vector_camera = (bpy.context.region_data.view_rotation @ V((0, 0, -1))) # get view direction
            # Get angle
            angle_to_camera = self.SignedAngle(up, direction_vector_camera, self.DirectionVector) * -1

            return self.ApplyRotationWithOffset(angle_to_camera, vertices, self.DirectionVector)
        return vertices
    
    def ShapeArrow(self, offset=1.0, flip_arrow = False, scale = 0.75):
        """Calculate vetices for arrow, return arrow vertices(by face 3 vert) and arrow contour(Loop)"""
        offset = self.GetForwardVector() * offset
        a_v1 = V((+0.0, +1.0, +0.0)) + offset
        a_v2 = V((-0.7, +0.0, +0.0)) + offset
        a_v3 = V((-0.3, +0.0, +0.0)) + offset
        a_v4 = V((-0.3, -2.0, +0.0)) + offset
        a_v5 = V((+0.3, -2.0, +0.0)) + offset
        a_v6 = V((+0.3, +0.0, +0.0)) + offset
        a_v7 = V((+0.7, +0.0, +0.0)) + offset

        arrow_faces =[  a_v1, a_v2, a_v3,
                        a_v3, a_v4, a_v5,
                        a_v3, a_v5, a_v6,
                        a_v1, a_v6, a_v7,
                        a_v1, a_v3, a_v6]
        contour_arrow = [a_v1, a_v2, a_v3, a_v4, a_v5, a_v6, a_v7]

        arrow_faces = self.ApplyScale(scale, arrow_faces)
        contour_arrow = self.ApplyScale(scale, contour_arrow)

        arrow_faces = self.RTM(arrow_faces, flip_current_forward = flip_arrow)
        contour_arrow = self.RTM(contour_arrow, flip_current_forward = flip_arrow)

        arrow_faces = self.Facing(arrow_faces)
        contour_arrow = self.Facing(contour_arrow)

        arrow_faces = self.ViewSize(arrow_faces)
        contour_arrow = self.ViewSize(contour_arrow)

        return arrow_faces, contour_arrow
    
    def ShapePlane(self, scale = 2, custom_direction = None, offset=gv0):
        p_v1 = V((+1.0, +0.0, +1.0)) + offset
        p_v2 = V((+1.0, +0.0, -1.0)) + offset
        p_v3 = V((-1.0, +0.0, -1.0)) + offset
        p_v4 = V((-1.0, +0.0, +1.0)) + offset

        p = [p_v1, p_v2,p_v3, p_v4]
        p = self.ApplyScale(scale, p)
        p = self.RTM(p,custom_direction)
        p = self.ViewSize(p)

        plane_faces = [ p[0], p[3], p[1],
                        p[1], p[2], p[3],]
        
        contour_plane = [p[0], p[1], p[2], p[3]]

        return plane_faces, contour_plane
    
    def ShapeGrid(self, scale = 2):
        """Return array lines grid by paire A and B (LINES)"""
        max = 6
        min = -6
        grid = []
        for h in range(-4, 5, 2):
            grid.append(V((min,0, h))/6)
            grid.append(V((max,0, h))/6)
            grid.append(V((h, 0,max))/6)
            grid.append(V((h, 0,min))/6)

        grid = self.ApplyScale(scale, grid)
        grid = self.RTM(grid)
        grid = self.ViewSize(grid)
        return grid
    
    def ShapeRing(self, start_draw_direction, radius=RingRadius, offset=RingWidth):
        """Retrun faces, outer_lines_loop, internal_lines_loop """
        # Get circle
        num_points = 360 // CirclePointDivider
        circle_points = self.GetCircle(radius, num_points)

        # Get outer curcle. Do that before transfomation because both rings need in original position
        outer_radius = radius + offset
        circle_outer_points = self.ApplyOffsetByNormal(outer_radius, circle_points)

        # Rotate vertext by matrix axis
        circle_points= self.RTM(circle_points)
        circle_outer_points= self.RTM(circle_outer_points)

        # Rotate ring to First mouse clic
        angl = self.SignedAngle((circle_outer_points[0] - circle_points[0]).normalized(), start_draw_direction, self.DirectionVector) * -1
        circle_points = self.ApplyRotationWithOffset(angl, circle_points, self.DirectionVector)
        circle_outer_points = self.ApplyRotationWithOffset(angl, circle_outer_points, self.DirectionVector)

        # Make fixed screen size
        circle_points = self.ViewSize(circle_points)
        circle_outer_points = self.ViewSize(circle_outer_points)

        # Make faces
        circle_faces = []
        GetPlusOne = lambda index: int(math.fmod(index + 1, num_points))
        for i in range(num_points):
            # face tris 1 
            circle_faces.append(circle_points[i])
            circle_faces.append(circle_points[GetPlusOne(i)])
            circle_faces.append(circle_outer_points[GetPlusOne(i)])
            # face tris 2
            circle_faces.append(circle_points[i])
            circle_faces.append(circle_outer_points[i])
            circle_faces.append(circle_outer_points[GetPlusOne(i)])

        return circle_faces, circle_outer_points, circle_points

    def GetCircle(self, radius, num_points, is_uv=False):
        coordinates = []
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            if not is_uv: 
                coordinates.append(V((x, 0.0, z)))  
            else:
                coordinates.append(V((x, z, 0.0)))
        return coordinates

    def MarkersForDegrees(self, DirectionVector, radius=RingRadius, Width=RingWidth):
        def MakeMarkers (offset, cell_count, segments):
            cell = Width / cell_count
            start_radius =  cell * offset + radius
            end_radius = (Width - (cell * offset)) + radius

            marks = self.GetCircle(start_radius, segments)
            outer_marks = self.ApplyOffsetByNormal(end_radius, marks)

            arr = []
            for i in range(segments):
                arr.append(marks[i])
                arr.append(outer_marks[i])
            return arr
        
        lines_5_deg  = MakeMarkers(4.5, 10, 72)
        lines_15_deg = MakeMarkers(3, 10, 24)
        lines_45_deg = MakeMarkers(2, 10, 8 )
        lines_90_deg = MakeMarkers(0, 10, 4 )

        markers = (lines_5_deg + lines_15_deg + lines_45_deg + lines_90_deg)

        markers= self.RTM(markers)
        markers = self.ViewSize(markers)


        # Rotate ring to First mouse clic
        SignedAngle= lambda v1, v2, axis: v1.angle(v2) * (1 if axis.dot(v1.cross(v2)) >= 0 else -1)
        angl = SignedAngle(lines_5_deg[0], DirectionVector, self.DirectionVector) * -1
        markers = self.ApplyRotationWithOffset(angl, markers, self.DirectionVector)

        return markers

    def GhostGizmo(self, axis,scale = 1,):
        # Make axis
        x_axis = gvx * scale
        y_axis = gvy * scale
        z_axis = gvz * scale

        #Rotate axin by matrix Global coord to Local
        x_axis = x_axis @ self.Matrix.inverted()
        y_axis = y_axis @ self.Matrix.inverted()
        z_axis = z_axis @ self.Matrix.inverted()

        # make lines
        line_x = [self.Pivot, x_axis + self.Pivot]
        line_y = [self.Pivot, y_axis + self.Pivot]
        line_z = [self.Pivot, z_axis + self.Pivot]

        # Make Fixed Scale axises 
        line_x = self.ViewSize(line_x)
        line_y = self.ViewSize(line_y)
        line_z = self.ViewSize(line_z)

        # Make plane vetice with offset. Commit it for see what will hapens
        p_v1 = V((+2.0, +0.0, +2.0)) + (V((0,0,-2)) if axis == 2 else (V((-2,0,0)) if axis == 0 else V((0,0,0))))
        p_v2 = V((+2.0, +0.0, -0.0)) + (V((0,0,-2)) if axis == 2 else (V((-2,0,0)) if axis == 0 else V((0,0,0))))
        p_v3 = V((-0.0, +0.0, -0.0)) + (V((0,0,-2)) if axis == 2 else (V((-2,0,0)) if axis == 0 else V((0,0,0))))
        p_v4 = V((-0.0, +0.0, +2.0)) + (V((0,0,-2)) if axis == 2 else (V((-2,0,0)) if axis == 0 else V((0,0,0))))
        plane = [p_v1, p_v2,p_v3, p_v4]
        plane = self.ApplyScale(scale/2, plane)

        # Rotate Plane by view vector
        dir = gvx if axis == 0 else (gvy if axis == 1 else gvz)
        angle = self.GetForwardVector().angle(dir)
        axis_for_rotation = self.GetForwardVector().cross(dir)
        plane = [i @ mathutils.Matrix.Rotation(angle, 4, axis_for_rotation) for i in plane]

        # Trasfer from Global to Local
        plane = [(i) @ self.Matrix.inverted() for i in plane]
        # Add pivot offset
        plane = [(i + self.Pivot) for i in plane]

        # Scale plane
        plane = self.ViewSize(plane)

        # Make faces and lines of plane
        plane_faces = [ plane[0], plane[3], plane[1],
                        plane[1], plane[2], plane[3],]
        contour_plane = [plane[0], plane[1], plane[2], plane[3]]

        return line_x, line_y, line_z, plane_faces, contour_plane

    def DrawTextInderMouse(self, mouse_pos, text):
        if Is_3d():
            region = bpy.context.region
            RegionView3D = bpy.context.region_data
            pivot_2d = view3d_utils.location_3d_to_region_2d(region, RegionView3D, self.Pivot)
            mouse_2d = view3d_utils.location_3d_to_region_2d(region, RegionView3D, mouse_pos)
        else:
            mouse_2d = mouse_pos
            pivot_2d = self.Pivot
            
        mouse_2d = (mouse_2d - pivot_2d).normalized() * 30.0 + mouse_2d
        text = f"{text:.1f}"+"°"
        blf.size(0, 20) 
        d = blf.dimensions(0, str(text))
        blf.position(0, mouse_2d.x - (d[0]/2), mouse_2d.y,0)


        blf.color(0,0.8,0.8,0.8,0.8)
        blf.draw(0, text)

    def BatchGizmo3D(self, direction, gizmo_scale=1.2):
        # Make shader and batsh
        line_x, line_y, line_z, plane_faces, plane_contour = self.GhostGizmo(direction, gizmo_scale)

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set("ALPHA")
        
        batch = batch_for_shader(shader, 'TRIS', {"pos": plane_faces})
        shader.uniform_float("color", AxisColor(direction, 0.5))
        batch.draw(shader)

        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.uniform_float("lineWidth", 5)
        shader.uniform_float("viewportSize", (bpy.context.area.width, bpy.context.area.height))

        shader.uniform_float("color", AxisColor(0, 0.5))
        batch = batch_for_shader(shader, 'LINES', {"pos": line_x})
        batch.draw(shader)
        shader.uniform_float("color", AxisColor(1, 0.5))
        batch = batch_for_shader(shader, 'LINES', {"pos": line_y})
        batch.draw(shader)
        shader.uniform_float("color", AxisColor(2, 0.5))
        batch = batch_for_shader(shader, 'LINES', {"pos": line_z})
        batch.draw(shader)



        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.uniform_float("lineWidth", 1)
        shader.uniform_float("color", (0.8, 0.8, 0.8, 1.0))
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": plane_contour})
        batch.draw(shader)
        pass
    
    def BatchGizmo2D(self, pivot):
        make_lines = lambda vector: [pivot, vector+pivot]
        x = V((100,0)).to_3d()
        y = V((0,100)).to_3d()


        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.uniform_float("viewportSize", (bpy.context.area.width, bpy.context.area.height))
        gpu.state.blend_set("ALPHA")
        shader.uniform_float("lineWidth", 2)

        shader.uniform_float("color", AxisColor(0, 0.8))
        batch = batch_for_shader(shader, 'LINES', {"pos": make_lines(x)})
        batch.draw(shader)

        shader.uniform_float("color", AxisColor(1, 0.8))
        batch = batch_for_shader(shader, 'LINES', {"pos": make_lines(y)})
        batch.draw(shader)

    def BatchMirror_Zero(self, arrow_offset, scale = 0.75):
            
            arrow_faces, contur_arrow = self.ShapeArrow(arrow_offset, flip_arrow=False, scale=scale)
            arrow_2_faces, contur_2_arrow = self.ShapeArrow(arrow_offset, flip_arrow=True, scale=scale)
            if Is_3d():
                plane_faces, contur_plane = self.ShapePlane()
                grid = self.ShapeGrid()

            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set("ALPHA")

            if Is_3d():
                batch = batch_for_shader(shader, 'TRIS', {"pos": plane_faces})
                shader.uniform_float("color", (0.0, 0.3, 0.6, 0.3))
                batch.draw(shader)

            batch = batch_for_shader(shader, 'TRIS', {"pos": arrow_faces})
            shader.uniform_float("color", (0.0, 0.3, 0.6, 0.3))
            batch.draw(shader)

            batch = batch_for_shader(shader, 'TRIS', {"pos": arrow_2_faces})
            shader.uniform_float("color", (0.0, 0.3, 0.6, 0.3))
            batch.draw(shader)

            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
            shader.uniform_float("lineWidth", 3)
            shader.uniform_float("viewportSize", (bpy.context.area.width, bpy.context.area.height))
            if Is_3d():
                batch = batch_for_shader(shader, 'LINES', {"pos": grid})
                shader.uniform_float("color", (1.0, 1.0, 1.0, 0.15))
                batch.draw(shader)

                batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_plane})
                shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
                batch.draw(shader)
            else: shader.uniform_float("lineWidth", 1.5)
            
            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_arrow})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
            batch.draw(shader)

            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_2_arrow})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
            batch.draw(shader)

            if not Is_3d():
                side = lambda axis, dir = 1: (V((0,200,0)) if axis == 0 else V((200,0,0))) * dir + self.Pivot
                lines = [self.Pivot, side(self.AxisInMatrix), self.Pivot, side(self.AxisInMatrix, -1)]

                shader.uniform_float("lineWidth", 7)
                batch = batch_for_shader(shader, 'LINES', {"pos": lines})
                shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
                batch.draw(shader)

                shader.uniform_float("lineWidth", 5.0)
                batch = batch_for_shader(shader, 'LINES', {"pos": lines})
                shader.uniform_float("color", AxisColor(2,0.8))
                batch.draw(shader)

    def BatchRotation(self, start_drawing_vector, normal, angle, povit, current_mouse_pos):
        if Is_3d():
            radius = RingRadius
            offset = RingWidth
        else:
            radius = 80
            offset = 20

        ring_faces, outer_contour, inner_contour = self.ShapeRing(start_drawing_vector, radius, offset)
        markers_deg = self.MarkersForDegrees(start_drawing_vector, radius, offset)


        # Slices for visual fiilng the distance traveled
        val = int(math.fmod(angle*-1 // CirclePointDivider, 360//CirclePointDivider)) *-1 * 6 # Step filling faces. * 6 because we have 6 vertices per step
        ring_faces_empty = (ring_faces[:val] if angle <= 0 else ring_faces[val:])  if val != 0 else ring_faces
        ring_faces_fill = (ring_faces[val:] if angle < 0 else ring_faces[:val])

        # axis = GetIndexOfMaxValueInVector(normal)
        color = AxisColor(self.AxisInMatrix)

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set("ALPHA")

        batch = batch_for_shader(shader, 'TRIS', {"pos": ring_faces_fill})
        shader.uniform_float("color", color)
        batch.draw(shader)
        batch = batch_for_shader(shader, 'TRIS', {"pos": ring_faces_empty})
        shader.uniform_float("color", (0.5, 0.5, 0.5, 0.3))
        batch.draw(shader)


        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.uniform_float("lineWidth", 1)
        shader.uniform_float("viewportSize", (bpy.context.area.width, bpy.context.area.height))

        shader.uniform_float("color", (0.2, 0.2, 0.2, 1.0))
        batch = batch_for_shader(shader, 'LINES', {"pos": markers_deg})
        batch.draw(shader)

        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
        batch = batch_for_shader(shader, 'LINES', {"pos": [povit, start_drawing_vector]})
        batch.draw(shader)
        batch = batch_for_shader(shader, 'LINES', {"pos": [povit, current_mouse_pos]})
        batch.draw(shader)

        shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": outer_contour})
        batch.draw(shader)
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": inner_contour})
        batch.draw(shader)

        # batch = batch_for_shader(shader, 'LINES', {"pos": [povit, self.Matrix.col[2].to_3d()*2+povit]})
        # batch.draw(shader)

class UserSettings():
    def __init__(self):
        if Is_3d():
            self.GetSnappingSettings()
            self.GetTransfromSettings()
        else:
            self.GetUVSettings()
        self.GetCursorSettings()
        self.GetUseDragImmediately()
        bpy.context.preferences.inputs.use_drag_immediately = True
        """Need the action to end when you release the mouse button"""
        self.UseReturnAllSettings = True
        """Use when opereaten should not return setting, for example Mirror and Scale Zero"""

    def GetSnappingSettings(self):
        self.snap_elements = bpy.context.scene.tool_settings.snap_elements
        self.snap_target = bpy.context.scene.tool_settings.snap_target
        self.use_snap_align_rotation = bpy.context.scene.tool_settings.use_snap_align_rotation
        self.use_snap_grid_absolute = bpy.context.scene.tool_settings.use_snap_grid_absolute
        self.use_snap_backface_culling = bpy.context.scene.tool_settings.use_snap_backface_culling
        self.use_snap_selectable = bpy.context.scene.tool_settings.use_snap_selectable
        self.use_snap_translate = bpy.context.scene.tool_settings.use_snap_translate
        self.use_snap_rotate = bpy.context.scene.tool_settings.use_snap_rotate
        self.use_snap_scale = bpy.context.scene.tool_settings.use_snap_scale
        self.use_snap = bpy.context.scene.tool_settings.use_snap

    def GetCursorSettings(self):
        if not Is_3d(): self.CursorLocation = GetCursorPosition(); return 0
        self.CursorLocation = GetCursorPosition()
        self.CursoreRotation = bpy.context.scene.cursor.rotation_euler.copy()
    def GetUseDragImmediately(self):
        self.use_drag_immediately = bpy.context.preferences.inputs.use_drag_immediately
    def GetTransfromSettings(self):
        self.transform_orientation_slots = bpy.context.scene.transform_orientation_slots[0].type
        self.transform_pivot_point = bpy.context.scene.tool_settings.transform_pivot_point

    def SetCursorSettings(self):
        if not Is_3d(): SetCursorPosition(self.CursorLocation); return 0
        SetCursorPosition(self.CursorLocation)
        bpy.context.scene.cursor.rotation_euler = self.CursoreRotation

    def GetUVSettings(self):
        self.snap_uv_element = bpy.context.scene.tool_settings.snap_uv_element
        self.pivot_point = bpy.context.space_data.pivot_point
        self.snap_target = bpy.context.scene.tool_settings.snap_target

    def SetUVSettings(self):
        bpy.context.scene.tool_settings.snap_uv_element = self.snap_uv_element
        bpy.context.space_data.pivot_point = self.pivot_point
        bpy.context.scene.tool_settings.snap_target = self.snap_target

    def SetSnappingSettings(self):
        bpy.context.scene.tool_settings.snap_elements = self.snap_elements
        bpy.context.scene.tool_settings.snap_target = self.snap_target
        bpy.context.scene.tool_settings.use_snap_align_rotation = self.use_snap_align_rotation
        bpy.context.scene.tool_settings.use_snap_grid_absolute = self.use_snap_grid_absolute
        bpy.context.scene.tool_settings.use_snap_backface_culling = self.use_snap_backface_culling
        bpy.context.scene.tool_settings.use_snap_selectable = self.use_snap_selectable
        bpy.context.scene.tool_settings.use_snap_translate = self.use_snap_translate
        bpy.context.scene.tool_settings.use_snap_rotate = self.use_snap_rotate
        bpy.context.scene.tool_settings.use_snap_scale = self.use_snap_scale
        bpy.context.scene.tool_settings.use_snap = self.use_snap

        bpy.context.scene.cursor.location = self.CursorLocation 
        bpy.context.scene.cursor.rotation_euler = self.CursoreRotation
    def SetUseDragImmediately(self):
        bpy.context.preferences.inputs.use_drag_immediately = self.use_drag_immediately
    def SetTransfromSettings(self):
            bpy.context.scene.transform_orientation_slots[0].type = self.transform_orientation_slots
            bpy.context.scene.tool_settings.transform_pivot_point = self.transform_pivot_point


    def ReturnAllSettings(self):
        if self.UseReturnAllSettings:
            if Is_3d():
                self.SetSnappingSettings()
                self.SetTransfromSettings()
            else:
                self.SetUVSettings()
            self.SetCursorSettings()
            self.SetUseDragImmediately()

class ActionsState():
    """Actions State"""
    LeftMouse = False
    RightMouse = False
    MiddleMouse = False
    MoveMouse = False
    Space = False
    Shift = False
    Alt = False
    Ctrl = False
    Esc = False
    G = False
    X = False
    Y = False
    Z = False
    Pass = False

class ORI():
    """Operator Return Items"""
    RUNNING_MODAL = {'RUNNING_MODAL'}
    CANCELLED = {'CANCELLED'}
    FINISHED = {'FINISHED'}
    PASS_THROUGH = {'PASS_THROUGH'}
    INTERFACE = {'INTERFACE'}

#------------------BaseClass----------------------#
class AdvancedTransform(bpy.types.Operator):
    '''Base class for transform tools'''
    bl_idname = "transform.advancedtransform"
    bl_label = "Advanced Transform"

    def __init__(self):
        super().__init__()
        global Is_3d_var
        Is_3d_var = Is_3d()
        self.Toolname = "BaseClass"
        self._header = None
        self.SkipFrameValue = 4
        self.SkipFrameCurrent = 0
        self.OldMousePos = gv0
        self.NewMousePos = gv0
        self.Expected_Action = None
        """Function which be calling after delay"""
        self.PivotPoint = None
        self.TransfromOrientationMatrix = Matrix().to_3x3()
        self.ViewAxisInMatrix = None
        """index column for the best view direction"""
        self.NormalIntersectionPlane = gvz
        """Normal current view in transform orientation matrix"""
        self.CurrentDirectionAxis = None
        """Current axis 0, 1 or 2 """
        self.Event = bpy.types.Event
        """Temp Variable for saving event"""

        self.DrawGizmo = True

        self.ActionsState = ActionsState()
        self.ORI = ORI()
        self.UserSettings = UserSettings()
        self.GenerateLambdaConditions()
        self.GenerateDelegates()
        self.ShaderUtility = None
        if Is_3d():
            self.GML = lambda event: GetMouseLocation(self.PivotPoint,self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
        else:
            self.GML = lambda event: V((event.mouse_region_x, event.mouse_region_y)).to_3d()
        self.UpdateShaderUtilityARG = lambda: self.ShaderUtility.UpdateData(self.TransfromOrientationMatrix, self.PivotPoint, 2)
        """'Get Mouse Location' Just for convenience, to use the shorthand notation """

        self.position = None

        #self.ExludeAxisARG = lambda: (GetBestAxisInMatrix(self.TransfromOrientationMatrix,self.NormalIntersectionPlane ))

        #self.DrawCallBack_delegat = self.DrawCallBack # Empty drawcallback function

    def DrawCallBackBatch(self):
        if self.DrawGizmo:
            if Is_3d():
                if self.NormalIntersectionPlane != None and self.ViewAxisInMatrix != None:
                    self.ShaderUtility.BatchGizmo3D(self.ViewAxisInMatrix)
            else:
                self.ShaderUtility.BatchGizmo2D(self.PivotPoint)

    def UpdateDraw(self):
        if self.PivotPoint != None:
            if self.ShaderUtility == None:
                self.ShaderUtility = ShaderUtility(self.TransfromOrientationMatrix, self.PivotPoint, 2)
            else:
                self.UpdateShaderUtilityARG()
            if self.ShaderUtility != None:
                self.DrawCallBackBatch()

    def DrawCallBack2D(self, context):
        self.UpdateDraw()

    def DrawCallBack3D(self, context):
        self.UpdateDraw()


    def GenerateLambdaConditions(self):
        """Conditions for action, can be overridden at __init__ at children classes"""
        self.If_Modify = lambda event: event.shift or event.alt or event.ctrl
        self.If_Pass = lambda event: (self.If_MMove(event) and self.ActionsState.Pass) and self.SkipFrameCurrent != 0

        self.If_LM = lambda event: event.type == 'LEFTMOUSE'
        if not Is_3d(): self.If_LM = lambda event: event.type == 'LEFTMOUSE' and event.value == "PRESS"
        self.If_LM_Cond = lambda event: (self.If_LM(event) or self.ActionsState.LeftMouse) and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_MMove = lambda event: event.type == 'MOUSEMOVE' 
        self.If_MMove_Cond = lambda event: False

        self.If_MM = lambda event: event.type == 'MIDDLEMOUSE'
        self.If_MM_Cond = lambda event: self.If_MM(event) or self.ActionsState.MiddleMouse and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_RM = lambda event: event.type == 'RIGHTMOUSE'
        if not Is_3d(): self.If_RM = lambda event: event.type == 'RIGHTMOUSE' and event.value == "PRESS"
        self.If_RM_Cond = lambda event: self.If_RM(event) or self.ActionsState.RightMouse and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_Spcae = lambda event: event.type == 'SPACE'
        self.If_Spcae_Cond = lambda event: self.If_Spcae(event) or self.ActionsState.Space

        self.If_Shift = lambda event: event.shift
        self.If_Shift_Cond = lambda event: self.If_Shift(event)# and self.If_MMove(event)

        self.If_Alt = lambda event: event.alt
        self.If_Alt_Cond = lambda event: self.If_Alt(event)# and self.If_MMove(event)

        self.If_Ctrl = lambda event: event.ctrl
        self.If_Ctrl_Cond = lambda event: self.If_Ctrl(event) or self.ActionsState.Ctrl

        self.If_Esc = lambda event: event.type == 'ESC'
        self.If_Esc_Cond = lambda event: self.If_Esc(event) or self.ActionsState.Esc

        self.If_G = lambda event: event.unicode == 'G' or event.unicode == 'g'
        self.If_X = lambda event: event.unicode == 'X' or event.unicode == 'x'
        self.If_Y = lambda event: event.unicode == 'Y' or event.unicode == 'y'
        self.If_Z = lambda event: event.unicode == 'Z' or event.unicode == 'z'

    def GenerateDelegates(self):
        self.LM_D = lambda event: self.CallAction(event, self.AfterLeftMouseAction, self.BedoreLeftMouseAction, use_delay=True)
        self.RM_D = lambda event: self.CallAction(event, self.AfterRightMouseAction, self.BeforeRightMouseAction)
        self.MM_D = lambda event: self.CallAction(event, self.AfterMiddleMouseAction, self.BeforeMiddleMouseAction)
        self.MoveM_D = lambda event: self.CallAction(event, self.AfterMoveMouseAction, self.BeforeMoveMouseAction)
        self.Space_D = lambda event: self.CallAction(event, self.AfterSpaceAction, self.BeforeSpaceAction)
        self.Shift_D = lambda event: self.CallAction(event, self.AfterShiftAction , self.BeforeShiftAction)
        self.Alt_D = lambda event : self.CallAction(event, self.AfterAltAction , self.BeforeAltAction)
        self.Ctrl_D = lambda event : self.CallAction(event, self.AfterCtrlAction , self.BeforeCtrlAction)

    def PovitDriver(self, pivot=False, orientation=False):
        if Is_3d(): self.PovitDriver3D(pivot, orientation)
        else: self.PovitDriver2D()

    def PovitDriver2D(self):
        x, y = self.NewMousePos[0], self.NewMousePos[1]
        SetCursorPosition(V(context.region.view2d.view_to_region(x, y)))
        bpy.ops.transform.translate('INVOKE_DEFAULT',orient_type='GLOBAL', orient_matrix_type='GLOBAL', mirror=False, snap=True, snap_elements={'VERTEX'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, cursor_transform=True, release_confirm=True)
        bpy.context.space_data.pivot_point = 'CURSOR'
        bpy.context.scene.tool_settings.snap_target = 'CENTER'

    def PovitDriver3D(self, pivot=False, orientation=False):
        """Use inside 'Before' finction with super()"""
        position, rotation = SpawnCursorByRaycast(self.OldMousePos, set_poisition=pivot, set_orientation=orientation)
        if position!= None:
            if pivot:
                #SetCursorPosition(position)
                bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'
                bpy.context.scene.tool_settings.snap_target = 'CENTER'
            if orientation:
                SetCursorPosition(position)
                bpy.context.scene.transform_orientation_slots[0].type = 'CURSOR'
        self.GetMainData()
        # else: bpy.ops.view3d.cursor3d()

    # We can use actions before delay and after delay
    def BedoreLeftMouseAction(self, event):
        self.ActionsState.LeftMouse = True
        self.OldMousePos = self.GML(event)
    def AfterLeftMouseAction(self, event):
        self.ActionsState.LeftMouse = False
        return self.ORI.RUNNING_MODAL
    def BeforeRightMouseAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.RightMouse = True
    def AfterRightMouseAction(self, event):
        self.ActionsState.RightMouse = False
        return self.ORI.RUNNING_MODAL
    def BeforeMiddleMouseAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.MiddleMouse = True
    def AfterMiddleMouseAction(self, event):
        self.ActionsState.MiddleMouse = False
        return self.ORI.RUNNING_MODAL
    def BeforeMoveMouseAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.MoveMouse = True
    def AfterMoveMouseAction(self, event):
        self.ActionsState.MoveMouse = False
        return self.ORI.RUNNING_MODAL
    def BeforeSpaceAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.Space = True
    def AfterSpaceAction(self, event):
        self.ActionsState.Space = False
        return self.ORI.RUNNING_MODAL
    def BeforeShiftAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.Shift = True
        self.PovitDriver(pivot=True)
        pass
    def AfterShiftAction(self, event):
        self.ActionsState.Shift = False

        return self.ORI.RUNNING_MODAL
    def BeforeAltAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.Alt = True
        # self.cursor_temp_value = GetCursorPosition()
        self.PovitDriver(orientation=True)
    def AfterAltAction(self, event):
        self.ActionsState.Alt = False
        # SetCursorPosition(self.cursor_temp_value)
        return self.ORI.RUNNING_MODAL
    def BeforeCtrlAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.Ctrl = True
    def AfterCtrlAction(self, event):
        self.ActionsState.Ctrl = False
        return self.ORI.RUNNING_MODAL
    def Before_G(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.G = True
    def After_G(self, event):
        self.ActionsState.G = False
        return self.ORI.RUNNING_MODAL
    def Before_X(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.x = True
    def After_X(self, event):
        self.ActionsState.x = False
        return self.ORI.RUNNING_MODAL
    def Before_Y(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.Y = True
    def After_Y(self, event):
        self.ActionsState.Y = False
        return self.ORI.RUNNING_MODAL
    def Before_Z(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.Z = True
    def After_Z(self, event):
        self.ActionsState.Z = False
        return self.ORI.RUNNING_MODAL

    def Pass(self, event):
        pass

    def SaveEvent(self, event):
        self.Event.alt = event.alt
        self.Event.ascii = event.ascii
        self.Event.ctrl = event.ctrl
        # if self.Event.direction:
        #     self.Event.direction = event.direction
        self.Event.is_consecutive = event.is_consecutive
        self.Event.is_mouse_absolute = event.is_mouse_absolute
        self.Event.is_repeat = event.is_repeat
        self.Event.is_tablet = event.is_tablet
        self.Event.mouse_prev_press_x = event.mouse_prev_press_x
        self.Event.mouse_prev_press_y = event.mouse_prev_press_y
        self.Event.mouse_prev_x = event.mouse_prev_x
        self.Event.mouse_prev_y = event.mouse_prev_y
        self.Event.mouse_region_x = event.mouse_region_x
        self.Event.mouse_region_y = event.mouse_region_y
        self.Event.mouse_x = event.mouse_x
        self.Event.mouse_y = event.mouse_y
        self.Event.oskey = event.oskey
        self.Event.pressure = event.pressure
        self.Event.shift = event.shift
        self.Event.tilt = event.tilt
        self.Event.type = event.type
        self.Event.type_prev = event.type_prev
        self.Event.unicode = event.unicode
        self.Event.value = event.value
        self.Event.value_prev = event.value_prev
        self.Event.xr = event.xr

    def GetMainData(self):
        if Is_3d():
            self.PivotPoint = GetPivotPointPoistion()
            self.TransfromOrientationMatrix = GetTransfromOrientationMatrix()
            self.NormalIntersectionPlane = GetNormalForIntersectionPlane(self.TransfromOrientationMatrix)
            self.ViewAxisInMatrix = GetBestAxisInMatrix(self.TransfromOrientationMatrix, self.NormalIntersectionPlane)
        else:
            self.PivotPoint = GetPivotPointPoistion()
            deb(self.PivotPoint)

    def SetHeader(self, fun):
        if fun != None:
            _header = bpy.types.STATUSBAR_HT_header.draw
            bpy.types.STATUSBAR_HT_header.draw = fun
            return _header

    def AdditionalSetup(self, event):
        pass

    def SetUp(self, event):
        self._header = self.SetHeader(self._header)
        self.GetMainData()
        self.AdditionalSetup(event)
        # UI
        if Is_3d():
            self._handle_3d = bpy.types.SpaceView3D.draw_handler_add(self.DrawCallBack3D, (context, ), 'WINDOW','POST_VIEW')
            self._handle_2d = bpy.types.SpaceView3D.draw_handler_add(self.DrawCallBack2D, (context, ), 'WINDOW','POST_PIXEL')
        else:
            self.report({'INFO'}, "") # Need for update status bar
            self._handle_2d = bpy.types.SpaceImageEditor.draw_handler_add(self.DrawCallBack2D, (context, ), 'WINDOW','POST_PIXEL')
        
    def AdditionalExit(self):
        pass
    def Exit(self):
        deb("Exit")
        self.AdditionalExit()
        try:
            if Is_3d():
                if self._handle_3d: bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, 'WINDOW')
                if self._handle_2d: bpy.types.SpaceView3D.draw_handler_remove(self._handle_2d, 'WINDOW')
            else:
                if self._handle_2d: bpy.types.SpaceImageEditor.draw_handler_remove(self._handle_2d, 'WINDOW')
        except:
            pass
        SpawnCursorByRaycast("",free_meshes=True)
        self.SetHeader(self._header)
        bpy.context.area.header_text_set(None)
        self.UserSettings.ReturnAllSettings()
        bpy.context.area.tag_redraw()
    def Canceled(self):
        self.Exit()
        return self.ORI.CANCELLED
    def Finish(self):
        self.Exit()
        return self.ORI.FINISHED

    def ModalReturnDec(self, value):
        """ because __del__ not stable and operator execute() doesn't work i need decorator for exit from modal ¯\_(ツ)_/¯ """
        if   value == self.ORI.RUNNING_MODAL: return self.ORI.RUNNING_MODAL
        elif value == self.ORI.PASS_THROUGH: return self.ORI.PASS_THROUGH
        elif value == self.ORI.INTERFACE: return self.ORI.INTERFACE
        elif value == self.ORI.FINISHED: return self.Finish()
        elif value == self.ORI.CANCELLED: return self.Canceled()

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "VIEW_3D" or context.space_data.type == "IMAGE_EDITOR"

    def CallAction(self, event, action_after_delay, action_before_delay, use_delay = False):
        if use_delay:
            self.SkipFrameCurrent = self.SkipFrameValue
        self.GetMainData()

        self.ActionsState.Pass = use_delay

        if action_before_delay:
            action_before_delay(event)

        self.SaveEvent(event)

        if action_after_delay:
            self.Expected_Action = action_after_delay

        if not use_delay:
            if self.Expected_Action:
                return self.Expected_Action(event)
            else: return self.ORI.RUNNING_MODAL
        else:
            return self.ModalReturnDec(self.ORI.RUNNING_MODAL)

    def modal(self, context, event):

        if self.If_Esc(event): return self.ModalReturnDec(self.Canceled())
        # -----------------------Skip Frames-------------------------------------------------------------#
        if self.If_Pass(event):
            self.SkipFrameCurrent -= 1
            return self.ORI.RUNNING_MODAL
        elif self.Expected_Action and self.ActionsState.Pass and self.SkipFrameCurrent <= 0: 
            self.ActionsState.Pass = False
            self.NewMousePos = self.GML(event)
            return self.ModalReturnDec(self.Expected_Action(self.Event))
        
        # -----------------------Actions---------------------------------------------------#
        if self.If_LM_Cond(event): return self.ModalReturnDec(self.LM_D(event))
        if self.If_RM_Cond(event): return self.ModalReturnDec(self.RM_D(event))
        if self.If_MM_Cond(event): return self.ModalReturnDec(self.MM_D(event))
        if self.If_Spcae_Cond(event): return self.ModalReturnDec(self.Space_D(event))
        if self.If_Shift_Cond(event): return self.ModalReturnDec(self.Shift_D(event))
        if self.If_Alt_Cond(event): return self.ModalReturnDec(self.Alt_D(event))
        if self.If_Ctrl_Cond(event): return self.ModalReturnDec(self.Ctrl_D(event))
        if self.If_MMove_Cond(event): return self.ModalReturnDec(self.MoveM_D(event))

        if self.If_G(event): return self.ModalReturnDec(self.CallAction(event, self.After_G, self.Before_G))
        if self.If_X(event): return self.ModalReturnDec(self.CallAction(event, self.After_X, self.Before_X))
        if self.If_Y(event): return self.ModalReturnDec(self.CallAction(event, self.After_Y, self.Before_Y))
        if self.If_Z(event): return self.ModalReturnDec(self.CallAction(event, self.After_Z, self.Before_Z))
        
        return self.ORI.RUNNING_MODAL

    def CheckSelection(self):
        """If something selected cursor chenge position"""
        rand_pos = V((-51651556780912,51651651651,0))
        selections = False
        CL = GetCursorPosition()
        SetCursorPosition(rand_pos)
        SetCursorToSelection()
        if GetCursorPosition() != rand_pos: selections = True
        SetCursorPosition(CL)
        return selections

    def invoke(self, context, event):
        self.SetUp(event)
        if self.CheckSelection():  
            context.window_manager.modal_handler_add(self)
            return self.ORI.RUNNING_MODAL
        else: 
            print("Cansel", self.Toolname)
            return self.ORI.CANCELLED

#------------------Transforms------------------#
class AdvancedMove(AdvancedTransform):
    ''' Advanced move '''
    bl_idname = "transform.advanced_move"
    bl_label = "Advanced Move"

    def __init__(self):
        super().__init__()
        self.toolName = "Advanced Move"
        self._header = Headers.MoveHeader

    def AfterLeftMouseAction(self, event):
        # axis = GetMouseDirectionAxis(self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        axis = GetMouseDirectionAxis(self.PivotPoint ,self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        SetConstarin.SetMoveOnlyOneAxis(axis)
        return self.ORI.FINISHED
    
    def AfterRightMouseAction(self, event):
        SetConstarin.SetMoveExclude(self.ViewAxisInMatrix)
        return self.ORI.FINISHED
    
    @is_3d_required
    def AfterMiddleMouseAction(self, event):
        SetConstarin.SetMoveNoConstrainNoSnap()
        return self.ORI.FINISHED
    
    @is_3d_required
    def AfterSpaceAction(self, event):
        bpy.context.scene.tool_settings.snap_elements = {'FACE'}
        if bpy.context.scene.tool_settings.snap_target != 'CURSOR':
            bpy.context.scene.tool_settings.snap_target = 'CENTER'
        bpy.context.scene.tool_settings.use_snap_align_rotation = True
        # bpy.context.scene.tool_settings.snap_elements_individual = {'FACE_PROJECT'}
        bpy.context.scene.tool_settings.use_snap = True
        QuickMoveToMouse(self.OldMousePos)
        SetConstarin.SetMoveNoConstrain()
        return self.ORI.FINISHED

    @is_3d_required
    def After_G(self, event):
        if Is_EditMesh():
            if bpy.context.tool_settings.mesh_select_mode[1]:
                bpy.ops.transform.edge_slide('INVOKE_DEFAULT')
            else:
                bpy.ops.transform.vert_slide("INVOKE_DEFAULT")
        return self.ORI.FINISHED

    
    def After_X(self, event):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
        return self.ORI.FINISHED
    def After_Y(self, event):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
        return self.ORI.FINISHED
    
    @is_3d_required
    def After_Z(self, event):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
        return self.ORI.FINISHED

class AdvancedScale(AdvancedTransform):
    ''' Advanced Scale '''
    bl_idname = "transform.advanced_scale"
    bl_label = "Advanced Scale"

    def __init__(self):
        super().__init__()
        self._header = Headers.ScaleHeader
        self.Toolname = "Advanced Scale"

    def AfterLeftMouseAction(self, event):
        axis = GetMouseDirectionAxis(self.PivotPoint, self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        SetConstarin.SetScaleOnly(axis)
        return self.ORI.FINISHED
    def AfterRightMouseAction(self, event):
        self.GetMainData()
        SetConstarin.SetScaleExclude(self.ViewAxisInMatrix)
        return self.ORI.FINISHED
    
    def AfterSpaceAction(self, event):
        UserSettings = id(self.UserSettings)
        bpy.ops.transform.advanced_scale_zero('INVOKE_DEFAULT')
        return {'FINISHED'}
    
    @is_3d_required
    def AfterMiddleMouseAction(self, event):
        SetConstarin.SetScaleNoConstrain()
        return self.ORI.FINISHED

    def AfterCtrlAction(self, event):
        UserSettings = id(self.UserSettings)
        bpy.ops.transform.advanced_scale_mirror('INVOKE_DEFAULT')
        return self.ORI.FINISHED

    def After_X(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, False))
        return self.ORI.FINISHED
    def After_Y(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, False))
        return self.ORI.FINISHED
    @is_3d_required
    def After_Z(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, False, True))
        return self.ORI.FINISHED

class AdvancedScaleMirror(AdvancedTransform):
    ''' Advanced Scale '''
    bl_idname = "transform.advanced_scale_mirror"
    bl_label = "Advanced Scale Mirror"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self) -> None:
        super().__init__()
        self.Toolname = "Scale Mirror"
        self._header = None
        self.CurrentAxisForMirror = None
        self.If_MMove_Cond = lambda event: self.If_MMove(event)
        self.If_Ctrl_Cond = lambda event: not self.If_Ctrl(event) #(event.type == "LEFT_CTRL")# and event.value == "RELEASE")
        self.UpdateShaderUtilityARG = lambda: (self.ShaderUtility.UpdateData(self.TransfromOrientationMatrix, self.PivotPoint, self.CurrentAxisForMirror))
        self._header
        
        self.UserSettings.UseReturnAllSettings = False

        self.ScaleAction = SetConstarin.SetScaleMirror

        if Is_3d(): self.ARG = lambda:(self.CurrentAxisForMirror, self.TransfromOrientationMatrix.to_3x3(), self.PivotPoint)
        else: self.ARG = lambda:(self.CurrentAxisForMirror, self.TransfromOrientationMatrix.to_3x3(), V(bpy.context.region.view2d.region_to_view(self.PivotPoint.x,self.PivotPoint.y)).to_3d()) # we need in coor in region space instead of view

        if Is_3d(): self.ArrowOffset = -2.5
        else: self.ArrowOffset = -2.0
        if Is_3d(): self.ArrowScale = 0.75
        else: self.ArrowScale = 75

        self.GetMainDataOnce = True
        self.GetMouseAxis = lambda: GetMouseDirectionAxis(self.PivotPoint ,self.OldMousePos, self.PivotPoint, self.TransfromOrientationMatrix, force_multi_axis=True)

    def DrawCallBackBatch(self):
        if self.CurrentAxisForMirror != None:
            self.ShaderUtility.BatchMirror_Zero(self.ArrowOffset, self.ArrowScale)
    
    def AdditionalExit(self):
        if Is_EditMesh():
            bpy.ops.mesh.flip_normals()

    def GetMainData(self):
        if self.GetMainDataOnce:
            super().GetMainData()
            self.GetMainDataOnce = False

    def AdditionalSetup(self, event):
        self.OldMousePos = self.GML(event)
        self.CurrentAxisForMirror = self.GetMouseAxis

    def AfterCtrlAction(self, event):
        self.ScaleAction(*self.ARG())
        return self.ORI.FINISHED
    
    def AfterMoveMouseAction(self, event):
        self.CurrentAxisForMirror = self.GetMouseAxis()
        bpy.context.area.tag_redraw()
        return self.ORI.RUNNING_MODAL

class AdvancedScaleZero(AdvancedScaleMirror):
    ''' Advanced Scale '''
    bl_idname = "transform.advanced_scale_zero"
    bl_label = "Advanced Scale zero"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self) -> None:
        super().__init__()
        self.Toolname = "Scale Zero"
        self._header = None
        self.If_Spcae_Cond = lambda event: False
        self.If_Ctrl_Cond = lambda event: self.If_Spcae(event) and event.value == "RELEASE"
        self.ScaleAction = SetConstarin.SetScaleOnlySetZero
        self.ArrowOffset = 1.5

    def AdditionalExit(self):
        pass

class AdvancedRotation(AdvancedTransform):
    """ Advanced move """
    bl_idname = "transform.advanced_rotation"
    bl_label = "Advanced Rotation"
    bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        super().__init__()
        self._header = Headers.RotationHeader
        self.ToolName = 'Advanced Rotation'
        self.RotationValue = 0.0
        self.StartDrawVector = None
        self.LastAngle = gvx
        self.NormalIntersectionPlane = gvz
        self.RotationDirection = gvz

        self.LM_D = lambda event: self.CallAction(event, self.AfterLeftMouseAction, self.BedoreLeftMouseAction) # need for remove delay
        if not bool(get_addon_preferences().SwapMBForRotation):
            self.If_LM_Cond = lambda event: self.If_LM(event)
            self.If_RM_Cond = lambda event: self.If_RM(event) and (self.ActionsState.MoveMouse == False and self.ActionsState.LeftMouse == False)
        else:
            self.If_LM_Cond = lambda event: self.If_RM(event) 
            self.If_RM_Cond = lambda event: self.If_LM(event) and self.StartDrawVector == None
        self.If_MMove_Cond = lambda event: self.If_MMove(event) and (self.ActionsState.LeftMouse or self.ActionsState.Ctrl)
        self.If_Ctrl_Cond = lambda event: event.type == "LEFT_CTRL"
        self.AngleSnappingStep = int(get_addon_preferences().Snapping_Step)

        self.UpdateShaderUtilityARG = lambda: self.ShaderUtility.UpdateData(self.TransfromOrientationMatrix, self.PivotPoint, self.ViewAxisInMatrix)

        self.GetDirection = lambda v1: (v1 - self.PivotPoint).normalized()
    
    @is_3d_required
    def AdditionalSetup(self,event):
        bpy.context.scene.tool_settings.snap_elements_base = {'VERTEX', 'EDGE_MIDPOINT'}
        bpy.context.scene.tool_settings.use_snap_rotate = True

    def DrawCallBackBatch(self):
        super().DrawCallBackBatch()
        if self.StartDrawVector != None:# and self.NewMousePos.length() != 0:
            start_direction = self.GetDirection(self.StartDrawVector)
            normal = self.NormalIntersectionPlane
            angle = self.RotationValue
            pivot = self.PivotPoint
            self.ShaderUtility.BatchRotation(start_direction, normal, angle, pivot, self.NewMousePos)

    def DrawCallBack2D(self, context):
        super().DrawCallBack2D(context)
        if self.NewMousePos.length != 0:
            self.ShaderUtility.DrawTextInderMouse(self.NewMousePos, self.RotationValue)

        

    def InitialRotation(self, event):
        self.GetMainData()
        self.ActionsState.MoveMouse = True
        self.StartDrawVector = self.GML(event)
        self.NewMousePos = self.StartDrawVector.copy()
        self.LastAngle = self.GetDirection(self.NewMousePos)
        # self.ViewAxisInMatrix = GetBestAxisInMatrix(self.TransfromOrientationMatrix, GetViewDirection())
        # deb(self.ViewAxisInMatrix,"self.ViewAxisInMatrix")
        # self.RotationDirection = self.TransfromOrientationMatrix.col[self.ViewAxisInMatrix].to_3d()
        self.DrawGizmo = False

    def GetMainData(self):
        if self.StartDrawVector is None: # Disable update if we started rotation
            super().GetMainData()

    def AfterMiddleMouseAction(self, event):
        SetConstarin.SetRotationFree()
        return self.ORI.FINISHED
    
    def AfterSpaceAction(self, event):
        SetConstarin.trackball()
        return self.ORI.FINISHED
        
    def AfterLeftMouseAction(self, event):
        if event.value == 'PRESS':
            self.InitialRotation(event)
            return self.ORI.RUNNING_MODAL
        elif event.value == 'RELEASE':
            return self.ORI.FINISHED

    def AfterRightMouseAction(self, event):
        SetConstarin.SetRotationOnly(self.ViewAxisInMatrix)
        return self.ORI.FINISHED
    
    @is_3d_required 
    @is_edit_mesh
    def BeforeCtrlAction(self, event):
        if self.StartDrawVector is None:
            self.OldMousePos = self.GML(event)
          
    @is_3d_required 
    @is_edit_mesh
    def AfterCtrlAction(self, event):
        if event.type == "LEFT_CTRL" and event.value == "PRESS" and self.ActionsState.Ctrl == False:
            self.ActionsState.Ctrl = True
            self.InitialRotation(event) 
            self.SetHeader(Headers.RotationStepOn)
            self.obj = bpy.context.active_object
            self.bm = bmesh.from_edit_mesh(self.obj.data)
            self.SelectedVertices = [v for v in self.bm.verts if v.select]
            # normal_selected_vertices = [self.obj.matrix_world @ v.normal.copy() for v in self.SelectedVertices]
            pos_selected_vertices = [self.obj.matrix_world @ v.co.copy() for v in self.SelectedVertices]

            self.BaseNormal = MakeCustomTransformOrientation().col[2].to_3d()
            deb(self.BaseNormal,"self.BaseNormal")
            self.PlaneNormal = self.BaseNormal.copy()
            self.BaseOrigin = sum(pos_selected_vertices, V()) / len(pos_selected_vertices)

        if event.type == "LEFT_CTRL" and event.value == "RELEASE" and self.ActionsState.Ctrl == True:
            self.bm.free()
            return self.ORI.FINISHED

        return self.ORI.RUNNING_MODAL

    def BeforeMoveMouseAction(self, event):
        self.NewMousePos = self.GML(event)

    def AfterMoveMouseAction(self, event):
        bpy.context.area.tag_redraw()
        angle = self.LastAngle.angle(self.GetDirection(self.NewMousePos))
        angle = math.degrees(angle)
        self.Rotatate(angle)
        return self.ORI.RUNNING_MODAL

    def BeforeShiftAction(self, event):
        if self.ActionsState.MoveMouse == False:
            super().BeforeShiftAction(event)
        else:
            self.If_Shift_Cond = lambda event:  event.type == "LEFT_SHIFT" and event.value == "PRESS"
            if self.AngleSnappingStep == 1:
                step_value = int(get_addon_preferences().Snapping_Step)
                self.SetHeader(Headers.RotationStepOn)
                # Compensation rotation for step round
                angle = (self.RotationValue % step_value)
                angle = step_value - angle if angle > step_value / 2 else angle * -1
                
                self.Rotatate(angle)
                self.AngleSnappingStep = step_value
            else: 
                self.SetHeader(Headers.RotationStepOff)
                self.AngleSnappingStep = 1
            return self.ORI.RUNNING_MODAL


    def RotationConstrain(self):
        for v in self.SelectedVertices:
            intersection = intersect_line_plane(self.obj.matrix_world @ v.co, self.BaseNormal + (self.obj.matrix_world @ v.co), self.PivotPoint ,self.PlaneNormal, True)
            if intersection != None:
                v.co = self.obj.matrix_world.inverted() @ intersection
                bmesh.update_edit_mesh(self.obj.data)

    def Rotatate(self, angle):
        # Check rotation step
        if (angle != None and (round(angle / self.AngleSnappingStep) * self.AngleSnappingStep) != 0) or self.AngleSnappingStep == 1:
            if self.AngleSnappingStep != 1:
                angle = self.AngleSnappingStep
            else:
                angle = round(angle)
            
            rotate = lambda angle: mathutils.Matrix.Rotation(math.radians(angle), 3, self.NormalIntersectionPlane)

            # find third axis 1 is mouse direction 2 is view direction  and 3 (corss) look at pivot point
            cross=((self.GetDirection(self.NewMousePos) - self.GetDirection(self.OldMousePos))).normalized().cross(self.NormalIntersectionPlane)

            # if value biger then 0 counterclock-wise else clockwise
            pos_neg = self.GetDirection(self.NewMousePos).dot(cross) > 0
            angle = angle*-1 if pos_neg > 0 else angle

            self.RotationValue += angle
            # Rotate self.OldMousePos to current rotation
            self.OldMousePos = self.NewMousePos.copy()
            self.LastAngle = self.LastAngle @ rotate(angle)
            if self.ActionsState.LeftMouse:
                SetConstarin.SetRotationOnlyAT(angle*-1 , self.ViewAxisInMatrix)
            elif self.ActionsState.Ctrl:
                self.PlaneNormal = self.PlaneNormal @ rotate(angle)
                self.RotationConstrain()

addon_keymaps = []

def get_addon_preferences():
    ''' quick wrapper for referencing addon preferences '''
    addon_preferences = bpy.context.preferences.addons[__name__].preferences
    return addon_preferences

def get_hotkey_entry_item(km, kmi_value):
    """
    returns hotkey of specific type, with specific properties.name (keymap is not a dict, ref by keys is not enough
    if there are multiple hotkeys!)
    """
    for i, km_item in enumerate(km.keymap_items):
        if km.keymap_items[i].idname == kmi_value:
            return km_item
    return None

def add_hotkey():
    user_preferences = bpy.context.preferences
    addon_prefs = user_preferences.addons[__name__].preferences

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    km = kc.keymaps.new(name="3D View Generic", space_type='VIEW_3D', region_type='WINDOW')
    kmi = km.keymap_items.new(AdvancedMove.bl_idname, 'G', 'PRESS', shift=False, ctrl=False, alt=False)

    kmi.active = True
    addon_keymaps.append((km, kmi))

    wm1 = bpy.context.window_manager
    kc1 = wm1.keyconfigs.addon
    km1 = kc1.keymaps.new(name="3D View Generic", space_type='VIEW_3D', region_type='WINDOW')
    kmi1 = km1.keymap_items.new(AdvancedScale.bl_idname, 'S', 'PRESS', shift=False, ctrl=False, alt=False)

    kmi1.active = True
    addon_keymaps.append((km1, kmi1))

    wm2 = bpy.context.window_manager
    kc2 = wm2.keyconfigs.addon
    km2 = kc2.keymaps.new(name="3D View Generic", space_type='VIEW_3D', region_type='WINDOW')
    kmi2 = km2.keymap_items.new(AdvancedRotation.bl_idname, 'R', 'PRESS', shift=False, ctrl=False, alt=False)

    kmi2.active = True
    addon_keymaps.append((km2, kmi2))

class AdvancedTransform_Add_Hotkey(bpy.types.Operator):
    """ Add hotkey entry """
    bl_idname = "advanced_transform.add_hotkey"
    bl_label = "Advanced Transform Add Hotkey"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        add_hotkey()

        self.report({'INFO'}, "Hotkey added in User Preferences -> Input -> Screen -> Screen (Global)")
        return {'FINISHED'}

def remove_hotkey():
    """ clears all addon level keymap hotkeys stored in addon_keymaps """
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.user
    km = kc.keymaps['3D View Generic']

    for i in bpy.context.window_manager.keyconfigs.addon.keymaps['3D View Generic'].keymap_items:
        if i.name == 'Advanced Move' or i.name == 'VIEW3D_OT_advanced_rotation':
            bpy.context.window_manager.keyconfigs.addon.keymaps['3D View Generic'].keymap_items.remove(i)
        elif i.name == 'Advanced Scale' or i.name == 'VIEW3D_OT_advancedscale':
            bpy.context.window_manager.keyconfigs.addon.keymaps['3D View Generic'].keymap_items.remove(i)
        elif i.name == 'Advanced Rotation' or i.name == 'VIEW3D_OT_advancedmove':
            bpy.context.window_manager.keyconfigs.addon.keymaps['3D View Generic'].keymap_items.remove(i)

class AdvancedTransformPref(bpy.types.AddonPreferences):
    bl_idname = __name__

    Snapping_Step: EnumProperty(
        items=[('5', "5", ""),
               ('10', "10", ""),
               ('15', "15", ""),
               ('30', "30", ""),
               ('45', "45", ""),
               ('90', "90", "")
               ],
        description="",
        name="Rotation Snapping Step",
        default='15',
        # update=use_cashes
        # caches_valid = True
    ) # type: ignore
    # Use_Advanced_Transform: BoolProperty(
    #     name="Use Advanced Transform Rotation",
    #     default=False,
    #     description="Is not use standard blender rotation without snapping (left mouse button)."
    #                 "Standard has great performance.",
    # )

    MultiAxisDrag: BoolProperty(
        name="Multi Axis Drag (BETA)",
        default=False,
        description="You can do constrain by 3 axes",
    )
    SwapMBForRotation: BoolProperty(
        name="Swap LMB and RMB for rotation",
        default=False,
        description="Swap LMB and RMB for rotation",
    )


    def draw(self, context):
        layout = self.layout
        box0 = layout.box()
        row1 = box0.row()
        row2 = box0.row()
        row3 = box0.row()
        row4 = box0.row()
        row1.prop(self, "Snapping_Step")
        # row2.prop(self, "Use_Advanced_Transform")
        row3.prop(self, "MultiAxisDrag")
        row4.prop(self, "SwapMBForRotation")
        # ---------------------------------
        box = layout.box()
        split = box.split()
        col = split.column()

        col.separator()
        wm = bpy.context.window_manager
        kc = wm.keyconfigs.user
        km = kc.keymaps['3D View Generic']
        kmi = get_hotkey_entry_item(km, "transform.advanced_move")
        if kmi:
            col.context_pointer_set("keymap", km)
            rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)
        else:
            col.label("No hotkey entry found")
            col.operator(AdvancedTransform_Add_Hotkey.bl_idname, text="Add hotkey entry", icon='ZOOMIN')
        # -----------------------------------
        box1 = layout.box()
        split1 = box1.split()
        col1 = split1.column()

        col1.separator()
        wm1 = bpy.context.window_manager
        kc1 = wm1.keyconfigs.user
        km1 = kc1.keymaps['3D View Generic']
        kmi1 = get_hotkey_entry_item(km1, "transform.advanced_scale")
        if kmi1:
            col1.context_pointer_set("keymap", km1)
            rna_keymap_ui.draw_kmi([], kc1, km1, kmi1, col1, 0)
        else:
            col1.label("No hotkey entry found")
            col1.operator(AdvancedTransform_Add_Hotkey.bl_idname, text="Add hotkey entry", icon='ZOOMIN')
        # --------------------------------------
        box2 = layout.box()
        split2 = box2.split()
        col2 = split2.column()

        col2.separator()
        wm2 = bpy.context.window_manager
        kc2 = wm2.keyconfigs.user
        km2 = kc2.keymaps['3D View Generic']
        kmi2 = get_hotkey_entry_item(km2, "transform.advanced_rotation")
        if kmi2:
            col2.context_pointer_set("keymap", km2)
            rna_keymap_ui.draw_kmi([], kc2, km2, kmi2, col2, 0)
        else:
            col2.label("No hotkey entry found")
            col2.operator(AdvancedTransform_Add_Hotkey.bl_idname, text="Add hotkey entry", icon='ZOOMIN')

class SetConstarin():
    @staticmethod
    def SetupSingleAxis(axis):
        return (axis == 0, axis == 1, axis == 2)
    @staticmethod
    def SetupExcludeAxis(axis):
        return (not (axis == 0), not (axis == 1), not (axis == 2))
    @staticmethod
    def SetMoveOnlyOneAxis(axis):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis = SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetMoveExclude(axis):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=SetConstarin.SetupExcludeAxis(axis))
    @staticmethod
    def SetMoveNoConstrain():
        bpy.ops.transform.translate('INVOKE_DEFAULT')
    @staticmethod
    def SetMoveNoConstrainNoSnap():
        bpy.ops.transform.translate('INVOKE_DEFAULT')
    @staticmethod
    def SetRotationOnlyAT(value, axis):
        bpy.ops.transform.rotate(value=math.radians(value), constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetRotationOnly(axis):
        bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetRotationFree():
        bpy.ops.transform.rotate('INVOKE_DEFAULT')
    @staticmethod
    def trackball():
        bpy.ops.transform.trackball('INVOKE_DEFAULT')
    @staticmethod
    def SetScaleOnly(axis):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetScaleExclude(axis):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=SetConstarin.SetupExcludeAxis(axis))
    @staticmethod
    def SetScaleNoConstrain():
        bpy.ops.transform.resize('INVOKE_DEFAULT')
    @staticmethod
    def SetScaleOnlySetZero(axis, orient_matrix, center_override):
        axis_value = {0: (0.0, 1.0, 1.0), 1: (1.0, 0.0, 1.0), 2: (1.0, 1.0, 0.0)}
        bpy.ops.transform.resize(value=axis_value[axis],orient_matrix=orient_matrix,center_override=center_override, constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetScaleExcludeSetZero(axis):
        axis_value = {0: (0.0, 1.0, 1.0), 1: (1.0, 0.0, 1.0), 2: (1.0, 1.0, 0.0)}
        bpy.ops.transform.resize(value=axis_value[axis], constraint_axis=SetConstarin.SetupExcludeAxis[axis])
    @staticmethod
    def SetScaleMirror(axis, orient_matrix, center_override):
        axis_value = {0: (-1.0, 1.0, 1.0), 1: (1.0, -1.0, 1.0), 2: (1.0, 1.0, -1.0)}
        
        bpy.ops.transform.resize(value=axis_value[axis],orient_matrix=orient_matrix,center_override=center_override, constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SnapRotation(axis, value):
        bpy.ops.transform.rotate(value=math.radians(value),constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetMoveOnlyUV(axis):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetMoveExcludeUV():
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, True, False))


classes = (AdvancedMove, AdvancedScale, AdvancedRotation,
           AdvancedTransformPref, AdvancedScaleZero, AdvancedScaleMirror, AdvancedTransform_Add_Hotkey)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    add_hotkey()


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    remove_hotkey()


if __name__ == "__main__":
    register()
