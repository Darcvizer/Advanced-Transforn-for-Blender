# context.area: VIEW_3D
import math
import os

import copy
import blf
import bgl
import bmesh
import bpy
import gpu
import mathutils
import numpy as np
import rna_keymap_ui
from bpy import context
from bpy import context as C
from bpy.props import EnumProperty, BoolProperty
from bpy.props import IntProperty
from bpy.props import IntProperty, FloatProperty
from bpy.types import Header
from bpy.types import Operator
from bpy.types import Operator, Macro
from bpy.utils import register_class, unregister_class
from bpy_extras import view3d_utils
import bpy_extras 
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix
from mathutils import Vector, Matrix
from mathutils import Vector as V
from mathutils.geometry import intersect_line_plane, intersect_point_quad_2d, intersect_line_line_2d
from functools import partial

# KE MODs : changed RELEASE modal events to "if not event.is_repeat" in AdvMove op, (and in Scale)
# also some reworking in the modal LMB move
# and had to re-arrange a bit in LMB AdvScale op.
# Just placeholder stuff for basic functionality until better days for Vlad!
# 0.4 = blender 4.0 compat fixes

bl_info = {
    "name": "Advanced Transform 2",
    "location": "View3D > Advanced Transform 2",
    "description": "Advanced Transform 2",
    "author": "Vladislav Kindushov(Darcvizer)",
    "version": (0, 5, 0),
    "blender": (4, 1, 0),
    "category": "View3D"}

v0 = V((0,0,0))

DubegMode = True
CirclePointDivider = 3
RingWidth = 0.25
RingRadius = 1
AxisColor = lambda axis, alpha=0.3 : (0.8, 0.2, 0.2, alpha) if axis == 0 else ((0.1, 0.6, 0.1, alpha) if axis == 1 else (0.0, 0.3, 0.6, alpha))
def deb(value,dis=None, pos=None, dir=None, forced=False):
    """Debug function
    text: Value for print | It is general prtin()
    dis: sting = None | Discription for print
    pos: Vector = None |Position for created object
    dir: Vector = None Should be normalized!!!| Direction for created object
    forced: forced=False | Forced create object
    """
    def CreateObjects():
        bpy.ops.mesh.primitive_plane_add()
        plane = bpy.context.object
        #bpy.context.collection.objects.link(plane) 
        bpy.ops.object.empty_add()
        null = bpy.context.object
        #bpy.context.collection.objects.link(null)
        bpy.context.object.empty_display_type = 'SINGLE_ARROW'
        bpy.context.object.empty_display_size = 3   
        plane.location = pos if pos != None else Vector((0.0,0.0,0.0))
        plane.matrix_world = dir.to_track_quat('Z', 'Y').to_matrix().to_4x4() if dir != None else Vector((0.0,0.0,1.0)).to_track_quat('Z', 'Y').to_matrix().to_4x4()
        plane.name = "ADHelpPlane" 
        null.name = "ADHelpNull"
        bpy.ops.object.constraint_add(type='COPY_TRANSFORMS')
        bpy.context.object.constraints["Copy Transforms"].target = plane    
    if DubegMode:
        print(dis if dis != None and isinstance(dis, str) else "",': ' , value)
        plane = bpy.data.objects.get("ADHelpPlane")
        null = bpy.data.objects.get("ADHelpNull")
        if  pos != None or dir != None:
            if (plane and null):
                if plane and dir != None:
                    plane.matrix_world = dir.to_track_quat('Z', 'Y').to_matrix().to_4x4()
                if plane and pos != None:
                    plane.location = pos
            else:
                CreateObjects()
        if forced:
            CreateObjects()
    return 0

def GetBestAxisInMatrix(matrix, axis):
    """Find max similar axis in matrix"""
    index_best_axis = [abs(axis.dot(matrix.col[i])) for i in range(0,3)] # Get dot for all axes
    index_best_axis_index = index_best_axis.index(max(index_best_axis))
    return index_best_axis_index

def GetViewDirection():
    return bpy.context.region_data.view_rotation @ Vector((0, 0, -1))

def GetNormalForIntersectionPlane(TransfromOrientationMatrix):
    """Getting the closest axis to camera direction"""
    view_direction = GetViewDirection()
    index_best_axis_index = GetBestAxisInMatrix(TransfromOrientationMatrix, view_direction)
    normal_intersect_plane = (TransfromOrientationMatrix.col[index_best_axis_index] * -1).normalized()# do invert for normal because we need derection to camera
    return normal_intersect_plane.to_3d()

def GetPivotPointPoistion():
    # best wat to get point use cursor, because using loop if user selected 1000 meshes too slowly    
    condition = lambda name: bpy.context.scene.tool_settings.transform_pivot_point == name

    original_pivot_point = bpy.context.scene.cursor.location.copy()
    bpy.ops.view3d.snap_cursor_to_active() if condition('ACTIVE_ELEMENT') else bpy.ops.view3d.snap_cursor_to_selected() 
    new_pivot_point = bpy.context.scene.cursor.location.copy()
    bpy.context.scene.cursor.location = original_pivot_point
    return bpy.context.scene.cursor.location.copy() if condition('CURSOR') else new_pivot_point

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
            temp = bpy.context.scene.transform_orientation_slots[0].type
            bpy.ops.transform.create_orientation(name="AdvancedTransform", use_view=False, use=True,overwrite=True)
            if bpy.context.scene.transform_orientation_slots[0].type == "AdvancedTransform":
                matrix_transform_orientation = bpy.context.scene.transform_orientation_slots[0].custom_orientation.matrix.copy()
                bpy.ops.transform.delete_orientation()
                bpy.context.scene.transform_orientation_slots[0].type = temp
    elif condition('VIEW'):
        temp = bpy.context.scene.transform_orientation_slots[0].type
        bpy.ops.transform.create_orientation(name="AdvancedTransform", use_view=True, use=True, overwrite=True)
        if bpy.context.scene.transform_orientation_slots[0].type == "AdvancedTransform":
            matrix_transform_orientation = bpy.context.scene.transform_orientation_slots[0].custom_orientation.matrix.copy()
            bpy.ops.transform.delete_orientation()
            bpy.context.scene.transform_orientation_slots[0].type = temp
    elif condition('CURSOR'):
        matrix_transform_orientation = bpy.context.scene.cursor.rotation_euler.to_matrix().copy()
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

    loc = intersect_line_plane(ray_origin_mouse, view_vector_mouse, pivot_point, normal, True)
    return loc

def GetIndexOfMaxValueInVector(vector ):
    """Get axis by mouse direaction
    normal: vector
    return_index return name axis (x,y,z) or index (0,1,2)
    """
    index_of_max_value = max(range(len(vector)), key=lambda i: abs(vector[i])) # get index of maximal value at vector
    return index_of_max_value

def GetMouseDirection(point_from, point_to, matrix):
    """Get mouse direction by 2 saved points, also if we have specific transform orientation we have to rotate vector by orientation matrix
    point_from: Vector | mouse point 1
    point_to: Vector | mouse point 2
    matrix: Matrix | Matrix for rotation vector
    """
    direction = (matrix.to_3x3().inverted() @ (point_from - point_to).normalized()) # Rotate move direction to transform orientation matrix
    #direction = (matrix.to_3x3() @ (point_from - point_to).normalized()) * -1
    return GetIndexOfMaxValueInVector(direction)

def SetupAxisUV(self):
    x = abs(self.temp_loc_first[0] - self.temp_loc_last[0])
    y = abs(self.temp_loc_first[1] - self.temp_loc_last[1])

    if x > y:
        return 'x'
    else:
        return 'y'

def SpawnCursorByRaycast(mouse_position, set_poisition = False, set_orientation = False):
    """Use raycast find the closest element on face
    Return position and rotation(euler)"""
    context = bpy.context
    depsgraph = context.evaluated_depsgraph_get()
    region = context.region
    rv3d = context.region_data
    camera_location = context.region_data.view_matrix.inverted().to_translation()
    
    mouse_direction = (mouse_position - camera_location).normalized()

    mouse_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_position)
    RESULT, LOCATION, NORMAL, INDEX, OBJECT, MATRIX = context.scene.ray_cast(depsgraph, origin=camera_location, direction=mouse_direction)
    points = []
    normals = []
    if RESULT:
        deb(INDEX,"INDEX")
        for e in OBJECT.data.polygons[INDEX].edge_keys:
                # Append element directions for transform orientation, Translate wrld normal to local because function gizmo use local coordinates                
                normals.append((MATRIX @ OBJECT.data.vertices[e[0]].normal))
                normals.append((MATRIX @ OBJECT.data.vertices[e[1]].normal))
                normals.append((normals[-1] - normals[-2]).normalized())
                # Append element position for pivot point
                v1 = MATRIX @ OBJECT.data.vertices[e[0]].co
                v2 = MATRIX @ OBJECT.data.vertices[e[1]].co
                edge = (v1 + v2)/2
                points.append(v1)
                points.append(v2)
                points.append(edge)
        # append face normal
        normals.append(NORMAL)
        # append center of face
        center = sum(points, mathutils.Vector()) / len(points)
        points.append(center)
        # get index the closest element to mouse
        closest_index = points.index(min(points, key=lambda v: (v - LOCATION).length))

        direction = normals[closest_index].to_track_quat('Z', 'Y').to_matrix().to_4x4()

        return points[closest_index], direction.to_euler()
    else: return None , None

class ShaderUtility():
    def __init__(self, matrix: Matrix, pivot: Vector, axis: str):
        self.UpdateData(matrix, pivot, axis)

        self.GetForwardVector = lambda: V((0, -1, 0))# if self.Axis == 1 else V((0, 1, 0)) 
        self.ApplyScale = lambda scale, arr: [(i @ mathutils.Matrix.Scale(scale, 4)) for i in arr]
        self.ApplyRotationWithOffset = lambda angle, arr, axis: [(i - self.Pivot) @ mathutils.Matrix.Rotation(angle, 3, axis) + self.Pivot for i in arr]
        self.ApplyRotation = lambda angle, arr, axis: [(i) @ mathutils.Matrix.Rotation(angle, 3, axis) for i in arr]
        self.ApplyOffsetByNormal = lambda value, arr:  [value * (i - V((0,0,0))).normalized() for i in arr] 
        self.SignedAngle= lambda v1, v2, axis: v1.angle(v2) * (1 if axis.dot(v1.cross(v2)) >= 0 else -1)
        pass


    def UpdateData(self, matrix, pivot, axis):
        self.AxisInMatrix = axis
        self.Matrix = matrix
        self.Pivot = pivot.copy()
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

        vertices = [i @ matrix + self.Pivot  for i in vertices]
        return vertices

    def ViewSize(self, vertices):
        """Calculate screen size"""
        view_distance = bpy.context.region_data.view_distance

        
        lerp = lambda a, b, t: (1 - t) * a + t * b # Thanks laundmo for example https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56
        scale_factor = lerp(0 ,1,view_distance / 10)

        scale_matrix = mathutils.Matrix.Scale(scale_factor,3)
        return [((v-self.Pivot)@ scale_matrix)+self.Pivot for v in vertices]
    
    def Facing(self, vertices):
        # Get view camera location
        camera_location = bpy.context.region_data.view_matrix.inverted().to_translation()

        # Get Up Direction of shape
        v1=vertices[0]
        v2=vertices[1]
        v3=vertices[2]

        vec1 = v2 - v1
        vec2 = v3 - v1

        up =vec1.cross(vec2).normalized()

        # Get camera direction
        direction_vector_camera = (bpy.context.region_data.view_rotation @ Vector((0, 0, -1))) # get view direction
        # Get angle
        angle_to_camera = self.SignedAngle(up, direction_vector_camera, self.DirectionVector) * -1

        return self.ApplyRotationWithOffset(angle_to_camera, vertices, self.DirectionVector)
    
    def ShapeArrow(self, offset=1.0, flip_arrow = False):
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

        arrow_faces = self.ApplyScale(0.75, arrow_faces)
        contour_arrow = self.ApplyScale(0.75, contour_arrow)

        arrow_faces = self.RTM(arrow_faces, flip_current_forward = flip_arrow)
        contour_arrow = self.RTM(contour_arrow, flip_current_forward = flip_arrow)

        arrow_faces = self.Facing(arrow_faces, )
        contour_arrow = self.Facing(contour_arrow, )

        arrow_faces = self.ViewSize(arrow_faces)
        contour_arrow = self.ViewSize(contour_arrow)

        return arrow_faces, contour_arrow
    
    def ShapePlane(self, scale = 2, custom_direction = None, offset=V((0,0,0))):
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
        g_v1  = V((-4.0000/6, -0.0000, +6.0000/6))
        g_v2  = V((-4.0000/6, +0.0000, -6.0000/6))
        g_v3  = V((-2.0000/6, -0.0000, +6.0000/6))
        g_v4  = V((-2.0000/6, +0.0000, -6.0000/6))
        g_v5  = V((+0.0000/6, -0.0000, +6.0000/6))
        g_v6  = V((+0.0000/6, +0.0000, -6.0000/6))
        g_v7  = V((+2.0000/6, -0.0000, +6.0000/6))
        g_v8  = V((+2.0000/6, +0.0000, -6.0000/6))
        g_v9  = V((+4.0000/6, -0.0000, +6.0000/6))
        g_v10 = V((+4.0000/6, +0.0000, -6.0000/6))
        g_v11 = V((-6.0000/6, -0.0000, +4.0000/6))
        g_v12 = V((+6.0000/6, -0.0000, +4.0000/6))
        g_v13 = V((-6.0000/6, -0.0000, +2.0000/6))
        g_v14 = V((+6.0000/6, -0.0000, +2.0000/6))
        g_v15 = V((-6.0000/6, +0.0000, +0.0000/6))
        g_v16 = V((+6.0000/6, +0.0000, +0.0000/6))
        g_v17 = V((-6.0000/6, +0.0000, -2.0000/6))
        g_v18 = V((+6.0000/6, +0.0000, -2.0000/6))
        g_v19 = V((-6.0000/6, +0.0000, -4.0000/6))
        g_v20 = V((+6.0000/6, +0.0000, -4.0000/6))

        grid = [g_v1,g_v2,g_v3,g_v4,g_v5,g_v6,g_v7,g_v8,g_v9,g_v10,g_v11,g_v12,g_v13,g_v14,g_v15,g_v16,g_v17,g_v18,g_v19,g_v20]
        grid = self.ApplyScale(scale, grid)
        grid = self.RTM(grid)
        grid = self.ViewSize(grid)
        return grid
    
    def ShapeRing(self, start_draw_direction):
        # Get circle
        radius = RingRadius
        num_points = 360 // CirclePointDivider
        circle_points = self.GetCircle(radius, num_points)

        # Get outer curcle. Do that before transfomation because both rings need in original position
        outer_radius = radius + RingWidth
        circle_outer_points = self.ApplyOffsetByNormal(outer_radius, circle_points)

        # Rotate vertext by matrix axis
        circle_points= self.RTM(circle_points)
        circle_outer_points= self.RTM(circle_outer_points)

        # Rotate ring to First mouse clic
        angl = self.SignedAngle((circle_outer_points[0] - circle_points[0]).normalized(), start_draw_direction, self.DirectionVector) * -1
        circle_points = self.ApplyRotationWithOffset(angl, circle_points, self.DirectionVector) # self.DirectionVector
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

    def GetCircle(self, radius, num_points):
        coordinates = []
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            coordinates.append(V((x, 0.0, z)))  
        return coordinates

    def MarkersForDegrees(self, DirectionVector):
        def MakeMarkers (offset, cell_count, segments):
            cell = RingWidth / cell_count
            start_radius =  cell * offset + RingRadius
            end_radius = (RingWidth - (cell * offset)) + RingRadius

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
        x_axis = V((1,0,0)) * scale
        y_axis = V((0,1,0)) * scale
        z_axis = V((0,0,1)) * scale

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
        dir = V((1,0,0)) if axis == 0 else (V((0,1,0)) if axis == 1 else V((0,0,1)))
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
        region = bpy.context.region
        RegionView3D = bpy.context.region_data

        coor = (mouse_pos - self.Pivot).normalized() * 1.0 + mouse_pos
        mouse_2d = view3d_utils.location_3d_to_region_2d(region, RegionView3D, coor)

        blf.size(0, 20) 
        d = blf.dimensions(0, str(text))
        blf.position(0, mouse_2d.x - (d[0]/2), mouse_2d.y - (d[1]/2),0)

        blf.color(0,0.8,0.8,0.8,0.8)
        blf.draw(0, str(text)+"°")

class UserSettings():
    def __init__(self):
        self.GetSnappingSettings()
        self.GetUseDragImmediately()
        self.GetCursorSettings()
        self.GetTransfromSettings()
        bpy.context.preferences.inputs.use_drag_immediately = True
        """Need the action to end when you release the mouse button"""

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
        self.CursorLocation = bpy.context.scene.cursor.location
        self.CursoreRotation = bpy.context.scene.cursor.rotation_euler
    def GetUseDragImmediately(self):
        self.use_drag_immediately = bpy.context.preferences.inputs.use_drag_immediately
    def GetTransfromSettings(self):
        self.transform_orientation_slots = bpy.context.scene.transform_orientation_slots[0].type
        self.transform_pivot_point = bpy.context.scene.tool_settings.transform_pivot_point 



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
    def SetCursorSettings(self):
        bpy.context.scene.cursor.location = self.CursorLocation 
        bpy.context.scene.cursor.rotation_euler = self.CursoreRotation
    def SetUseDragImmediately(self):
        bpy.context.preferences.inputs.use_drag_immediately = self.use_drag_immediately
    def SetTransfromSettings(self):
        bpy.context.scene.transform_orientation_slots[0].type = self.transform_orientation_slots
        bpy.context.scene.tool_settings.transform_pivot_point = self.transform_pivot_point

    def ReturnAllSettings(self):
        self.SetUseDragImmediately()
        self.SetSnappingSettings()
        self.SetCursorSettings()
        self.SetTransfromSettings()

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

class AdvancedTransform(Operator):
    '''Base class for transform tools'''
    bl_idname = "view3d.advancedtransform"
    bl_label = "Advanced Transform"

    def __init__(self):
        self.Toolname = "BaseClass"
        self.header_text = ""
        self.SkipFrameValue = 4
        self.SkipFrameCurrent = 0
        self.OldMousePos = mathutils.Vector((0.0,0.0,0.0))
        self.NewMousePos = mathutils.Vector((0.0,0.0,0.0))
        self.Expected_Action = None
        """Function which be calling after delay"""
        self.PivotPoint = None
        self.TransfromOrientationMatrix = None
        self.ViewAxisInMatrix = None
        """index column for the best view direction"""
        self.NormalIntersectionPlane = None
        """Normal current view in transform orientation matrix"""
        self.CurrentDirectionAxis = None
        """Current axis 0, 1 or 2 """
        self.Event = bpy.types.Event
        """Temp Variable for saving event"""

        self.ActionsState = ActionsState()
        self.ORI = ORI()
        self.UserSettings = UserSettings()
        self.GenerateLambdaConditions()
        self.GenerateDelegates()
        self.ShaderUtility = None

        self.GML = lambda event: GetMouseLocation(self.PivotPoint,self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
        self.UpdateShaderUtilityARG = lambda: self.ShaderUtility.UpdateData(self.TransfromOrientationMatrix, self.PivotPoint,0)
        """'Get Mouse Location' Just for convenience, to use the shorthand notation """

        


        #self.DrawCallBack_delegat = self.DrawCallBack # Empty drawcallback function


    def DrawCallBackBatch(self):
        if self.NormalIntersectionPlane != None and self.ViewAxisInMatrix != None:
            line_x, line_y, line_z, plane_faces, plane_contour = self.ShaderUtility.GhostGizmo(self.ViewAxisInMatrix, 1.2)

            # Make shader and batsh
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set("ALPHA")
            
            batch = batch_for_shader(shader, 'TRIS', {"pos": plane_faces})
            shader.uniform_float("color", AxisColor(self.ViewAxisInMatrix, 0.5))
            batch.draw(shader)

            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
            shader.uniform_float("lineWidth", 5)

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
            
    def DrawCallBack2D(self, context):
        pass

    def DrawCallBack3D(self, context):
        try:
            if self.TransfromOrientationMatrix != None and self.PivotPoint != None:
                if self.ShaderUtility == None:
                    self.ShaderUtility = ShaderUtility(self.TransfromOrientationMatrix, self.PivotPoint, 0)
                else:
                    self.UpdateShaderUtilityARG()
                if self.ShaderUtility != None:
                    self.DrawCallBackBatch()
        except:
            #self.Canceled()
            pass

    def GenerateLambdaConditions(self):
        """Conditions for action, can be overridden at __init__ at children classes"""
        self.If_Modify = lambda event: event.shift or event.alt
        self.If_Pass = lambda event: (self.If_MMove(event) and self.ActionsState.Pass) and self.SkipFrameCurrent != 0

        self.If_LM = lambda event: event.type == 'LEFTMOUSE'
        self.If_LM_Cond = lambda event: (self.If_LM(event) or self.ActionsState.LeftMouse) and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_MMove = lambda event: event.type == 'MOUSEMOVE' 
        self.If_MMove_Cond = lambda event: False

        self.If_MM = lambda event: event.type == 'MIDDLEMOUSE'
        self.If_MM_Cond = lambda event: self.If_MM(event) or self.ActionsState.MiddleMouse and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_RM = lambda event: event.type == 'RIGHTMOUSE'
        self.If_RM_Cond = lambda event: self.If_RM(event) or self.ActionsState.RightMouse and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_Spcae = lambda event: event.type == 'SPACE'
        self.If_Spcae_Cond = lambda event: self.If_Spcae(event) or self.ActionsState.Space

        self.If_Shift = lambda event: event.shift
        self.If_Shift_Cond = lambda event: self.If_Shift(event) and self.If_MMove(event)

        self.If_Alt = lambda event: event.alt
        self.If_Alt_Cond = lambda event: self.If_Alt(event) and self.If_MMove(event)

        self.If_Ctrl = lambda event: event.ctrl
        self.If_Ctrl_Cond = lambda event: self.If_Ctrl(event) or self.ActionsState.Ctrl

        self.If_Esc = lambda event: event.type == 'ESC'
        self.If_Esc_Cond = lambda event: self.If_Esc(event) or self.ActionsState.Esc

        self.If_G = lambda event: event.unicode == 'G' or event.unicode == 'g'
        self.If_X = lambda event: event.unicode == 'X' or event.unicode == 'x'
        self.If_Y = lambda event: event.unicode == 'Y' or event.unicode == 'y'
        self.If_Z = lambda event: event.unicode == 'Z' or event.unicode == 'z'

    def GenerateDelegates(self):
        self.LM_D = lambda event: self.StartDelay(event, self.AfterLeftMouseAction, self.BedoreLeftMouseAction, use_delay=True)
        self.RM_D = lambda event: self.StartDelay(event, self.AfterRightMouseAction, self.BeforeRightMouseAction)
        self.MM_D = lambda event: self.StartDelay(event, self.AfterMiddleMouseAction, self.BeforeMiddleMouseAction)
        self.MoveM_D = lambda event: self.StartDelay(event, self.AfterMoveMouseAction, self.BeforeMoveMouseAction)
        self.Space_D = lambda event: self.StartDelay(event, self.AfterSpaceAction, self.BeforeSpaceAction)
        self.Shift_D = lambda event: self.StartDelay(event, self.AfterShiftAction , self.BeforeShiftAction)
        self.Alt_D = lambda event : self.StartDelay(event, self.AfterAltAction , self.BeforeAltAction)
        self.Ctrl_D = lambda event : self.StartDelay(event, self.AfterCtrlAction , self.BeforeCtrlAction)

    def PovitDriver(self, pivot=False, orientation=False):
        """Use inside 'Before' finction with super()"""
        position, rotation = SpawnCursorByRaycast(self.OldMousePos, set_poisition=True)
        if position!= None:
            if pivot:
                bpy.context.scene.cursor.location = position
                bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'
            if orientation:
                bpy.context.scene.cursor.location = position
                bpy.context.scene.cursor.rotation_euler = rotation
                bpy.context.scene.transform_orientation_slots[0].type = 'CURSOR'
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
    def AfterShiftAction(self, event):
        self.ActionsState.Shift = False
        return self.ORI.RUNNING_MODAL
    def BeforeAltAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.Alt = True
        self.PovitDriver(orientation=True)
    def AfterAltAction(self, event):
        self.ActionsState.Alt = False
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
        self.PivotPoint = GetPivotPointPoistion()
        self.TransfromOrientationMatrix = GetTransfromOrientationMatrix()
        self.NormalIntersectionPlane = GetNormalForIntersectionPlane(self.TransfromOrientationMatrix)
        self.ViewAxisInMatrix = GetBestAxisInMatrix(self.TransfromOrientationMatrix, self.NormalIntersectionPlane)

    def AdditionalSetup(self, event):
        pass
    def SetUp(self, event):
        self.GetMainData()
        self.AdditionalSetup(event)
        # UI
        self._handle_3d = bpy.types.SpaceView3D.draw_handler_add(self.DrawCallBack3D, (context, ), 'WINDOW','POST_VIEW')# POST_PIXEL # POST_VIEW
        self._handle_2d = bpy.types.SpaceView3D.draw_handler_add(self.DrawCallBack2D, (context, ), 'WINDOW','POST_PIXEL')

    def Exit(self):
        try:
            if self._handle_3d: bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, 'WINDOW')
            if self._handle_2d: bpy.types.SpaceView3D.draw_handler_remove(self._handle_2d, 'WINDOW')
        except:
            pass
        self.UserSettings.ReturnAllSettings()
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
        return context.space_data.type == "VIEW_3D" and len(context.selected_objects) > 0
    
    def StartDelay(self, event, action_after_delay, action_before_delay, use_delay = False):
        if use_delay:
            self.SkipFrameCurrent = self.SkipFrameValue
        self.PivotPoint = GetPivotPointPoistion()
        self.TransfromOrientationMatrix = GetTransfromOrientationMatrix()
        self.NormalIntersectionPlane = GetNormalForIntersectionPlane(self.TransfromOrientationMatrix)
        self.ViewAxisInMatrix = GetBestAxisInMatrix(self.TransfromOrientationMatrix, self.NormalIntersectionPlane)

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
            self.NewMousePos = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
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

        if self.If_G(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.G, self.After_G, self.Before_G))
        if self.If_X(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.X, self.After_X, self.Before_X))
        if self.If_Y(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.Y, self.After_Y, self.Before_Y))
        if self.If_Z(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.Z, self.After_Z, self.Before_Z))
        
        return self.ORI.RUNNING_MODAL

    def invoke(self, context, event):
        context.area.header_text_set(self.header_text)       
        if context.space_data.type == 'VIEW_3D':
            self.SetUp(event)
            context.window_manager.modal_handler_add(self)
            print(self.Toolname)
            return self.ORI.RUNNING_MODAL
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return self.ORI.CANCELLED

class AdvancedMove(AdvancedTransform):
    ''' Advanced move '''
    bl_idname = "view3d.advancedmove"
    bl_label = "Advanced Move"

    def __init__(self):
        super().__init__()
        self.header_text = 'Drag LMB constraint axis, RMB translate by two axis, MMB free translate, SPACE free translate with snap and rotate along normal, Shift tweak new selection, Alt Current selection'
        self.toolName = "Advanced Move"
        self.If_Alt_Cond =   lambda event:   self.If_Alt(event) and self.If_MMove(event)
        self.If_Shift_Cond = lambda event: self.If_Shift(event) and self.If_MMove(event)

    def AfterLeftMouseAction(self, event):
        axis = GetMouseDirection(self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        SetConstarin.SetMoveOnlyOneAxis(axis)
        return self.ORI.FINISHED
    
    def AfterRightMouseAction(self, event):
        SetConstarin.SetMoveExclude(GetIndexOfMaxValueInVector(self.NormalIntersectionPlane))
        return self.ORI.FINISHED
    
    def AfterMiddleMouseAction(self, event):
        SetConstarin.SetMoveNoConstrainNoSnap()
        return self.ORI.FINISHED

    def AfterSpaceAction(self, event):
        bpy.context.scene.tool_settings.snap_elements = {'FACE'}
        bpy.context.scene.tool_settings.snap_target = 'CENTER'
        bpy.context.scene.tool_settings.use_snap_align_rotation = True
        bpy.context.scene.tool_settings.snap_elements_individual = {'FACE_PROJECT'}
        bpy.context.scene.tool_settings.use_snap = True
        SetConstarin.SetMoveNoConstrain()
        return self.ORI.FINISHED
    
    def After_G(self, event):
        if context.mode == "EDIT_MESH":
            if context.tool_settings.mesh_select_mode[1]:
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
    def After_Z(self, event):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
        return self.ORI.FINISHED

class AdvancedScale(AdvancedTransform):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale"
    bl_label = "Advanced Scale"

    def __init__(self):
        super().__init__()
        self.header_text = 'Drag LMB constraint axis, RMB resize by two axis, MMB free resize, SHIFT mirror, SPACE flatten'
        self.Toolname = "Advanced Scale"
        self.If_Alt_Cond =   lambda event:   self.If_Alt(event) and self.If_MMove(event)
        self.If_Shift_Cond = lambda event: self.If_Shift(event) and self.If_MMove(event)

    def AfterLeftMouseAction(self, event):
        axis = GetMouseDirection(self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        SetConstarin.SetScaleOnly(axis)
        return self.ORI.FINISHED
    def AfterRightMouseAction(self, event):
        SetConstarin.SetScaleExclude(GetIndexOfMaxValueInVector(self.NormalIntersectionPlane))
        return self.ORI.FINISHED
    
    def AfterSpaceAction(self, event):
        bpy.ops.view3d.advancedscale_zero('INVOKE_DEFAULT')
        
        return {'FINISHED'}
    
    def AfterMiddleMouseAction(self, event):
        SetConstarin.SetScaleNoConstrain()
        return self.ORI.FINISHED

    def AfterCtrlAction(self, event):
        bpy.ops.view3d.advancedscale_mirror('INVOKE_DEFAULT')
        return self.ORI.FINISHED

    def After_X(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, False))
        return self.ORI.FINISHED
    def After_Y(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, False))
        return self.ORI.FINISHED
    def After_Z(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, False, True))
        return self.ORI.FINISHED

class AdvancedScaleMirror(AdvancedTransform):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale_mirror"
    bl_label = "Advanced Scale Mirror"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self) -> None:
        super().__init__()
        self.Toolname = "Scale Mirror"
        self.CurrentAxisForMirror = None
        self.If_MMove_Cond = lambda event: self.If_MMove(event)
        self.If_Ctrl_Cond = lambda event: (event.type == "LEFT_CTRL")# and event.value == "RELEASE")
        self.UpdateShaderUtilityARG = lambda: (self.ShaderUtility.UpdateData(self.TransfromOrientationMatrix, self.PivotPoint, self.CurrentAxisForMirror))

        self.ScaleAction = SetConstarin.SetScaleMirror
        self.ArrowOffset = -2.5

    def DrawCallBackBatch(self):
        if self.CurrentAxisForMirror != None:
            arrow_faces, contur_arrow = self.ShaderUtility.ShapeArrow(self.ArrowOffset, flip_arrow=False)
            arrow_2_faces, contur_2_arrow = self.ShaderUtility.ShapeArrow(self.ArrowOffset, flip_arrow=True)
            plane_faces, contur_plane = self.ShaderUtility.ShapePlane()
            grid = self.ShaderUtility.ShapeGrid()

            # Make shader and batsh
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            # gpu.state.depth_test_set('LESS_EQUAL')
            #gpu.state.depth_mask_set(True)
            gpu.state.blend_set("ALPHA")
            gpu.state.line_width_set(15)

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
            gpu.state.depth_test_set('NONE')
            shader.uniform_float("lineWidth", 3)

            batch = batch_for_shader(shader, 'LINES', {"pos": grid})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.15))
            batch.draw(shader)

            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_plane})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
            batch.draw(shader)
            
            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_arrow})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
            batch.draw(shader)

            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_2_arrow})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
            batch.draw(shader)
            gpu.state.depth_mask_set(False)

    def GetData(self, event):
        self.NewMousePos = self.GML(event)
        mouse_direction = (self.NewMousePos - self.PivotPoint).normalized()
        self.CurrentAxisForMirror = GetBestAxisInMatrix(self.TransfromOrientationMatrix, mouse_direction)

    def AdditionalSetup(self, event):
        self.GetData(event)

    def AfterCtrlAction(self, event):
        print("zero")
        self.ScaleAction(self.CurrentAxisForMirror)
        return self.ORI.FINISHED
    
    def AfterMoveMouseAction(self, event):
        self.GetData(event)

        #self.report({'INFO'},"") # Update modal and check shift event. Remove bag with wating new tick
        return self.ORI.PASS_THROUGH

class AdvancedScaleZero(AdvancedScaleMirror):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale_zero"
    bl_label = "Advanced Scale zero"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self) -> None:
        super().__init__()
        self.Toolname = "Scale Zero"
        self.If_Spcae_Cond = lambda event: False
        self.If_Ctrl_Cond = lambda event: self.If_Spcae(event) and event.value == "RELEASE"
        self.ScaleAction = SetConstarin.SetScaleOnlySetZero
        self.ArrowOffset = 1.5

class AdvancedRotation(AdvancedTransform):
    """ Advanced move """
    bl_idname = "view3d.advanced_rotation"
    bl_label = "Advanced Rotation"
    bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        super().__init__()
        self.header_text = 'LMB constraint view axis, RMB constraint view axis snap, MMB free rotate'
        self.ToolName = 'Advanced Rotation'
        self.RotationValue = 0.0
        self.StartDrawVector = None
        self.LastAngle = Vector((1,0,0))

        self.If_Alt_Cond =   lambda event:   self.If_Alt(event) and self.If_MMove(event)
        self.If_Shift_Cond = lambda event: self.If_Shift(event) and self.If_MMove(event)
        self.LM_D = lambda event: self.StartDelay(event, self.AfterLeftMouseAction, self.BedoreLeftMouseAction)
        self.If_LM_Cond = lambda event: self.If_LM(event)# and (self.ActionsState.LeftMouse == False and self.ActionsState.MoveMouse == False)
        self.If_RM_Cond = lambda event: self.If_RM(event) and (self.ActionsState.MoveMouse == False and self.ActionsState.LeftMouse == False)
        self.If_MMove_Cond = lambda event: self.If_MMove(event) and (self.ActionsState.LeftMouse or self.ActionsState.RightMouse)
        self.AngleSnappingStep = int(context.preferences.addons[__name__].preferences.Snapping_Step)

        self.UpdateShaderUtilityARG = lambda: self.ShaderUtility.UpdateData(self.TransfromOrientationMatrix, self.PivotPoint, GetBestAxisInMatrix(self.TransfromOrientationMatrix, GetViewDirection()))

        self.GetDirection = lambda v1: (v1 - self.PivotPoint).normalized()

    def AdditionalSetup(self,event):
        bpy.context.scene.tool_settings.snap_elements_base = {'VERTEX', 'EDGE_MIDPOINT'}
        bpy.context.scene.tool_settings.use_snap_rotate = True

    def DrawCallBackBatch(self):
        if self.StartDrawVector != None:# and self.NewMousePos.length() != 0:
            start_drawing_vector = self.GetDirection(self.StartDrawVector)## + self.PivotPoint
            ring_faces, outer_contour, inner_contour = self.ShaderUtility.ShapeRing(start_drawing_vector)
            markers_deg = self.ShaderUtility.MarkersForDegrees(start_drawing_vector)

            # Slices for visual fiilng the distance traveled
            val = int(math.fmod(self.RotationValue // CirclePointDivider, 360//CirclePointDivider)) *-1 * 6 # Step filling faces. * 6 because we have 6 vertices per step
            ring_faces_fill = (ring_faces[:val] if self.RotationValue <= 0 else ring_faces[val:])
            ring_faces_empty = (ring_faces[val:] if self.RotationValue < 0 else ring_faces[:val]) if val != 0 else ring_faces

            axis = GetIndexOfMaxValueInVector(self.NormalIntersectionPlane)
            color = AxisColor(axis)

            # Make shader and batsh
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            #gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.blend_set("ALPHA")

            batch = batch_for_shader(shader, 'TRIS', {"pos": ring_faces_fill})
            shader.uniform_float("color", color)
            batch.draw(shader)
            batch = batch_for_shader(shader, 'TRIS', {"pos": ring_faces_empty})
            shader.uniform_float("color", (0.5, 0.5, 0.5, 0.3))
            batch.draw(shader)

            gpu.state.depth_test_set('ALWAYS')
            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
            shader.uniform_float("lineWidth", 1)

            shader.uniform_float("color", (0.2, 0.2, 0.2, 1.))
            batch = batch_for_shader(shader, 'LINES', {"pos": markers_deg})
            batch.draw(shader)

            shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": outer_contour})
            batch.draw(shader)
            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": inner_contour})
            batch.draw(shader)

            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
            batch = batch_for_shader(shader, 'LINES', {"pos": [self.PivotPoint, self.StartDrawVector]})
            batch.draw(shader)
            batch = batch_for_shader(shader, 'LINES', {"pos": [self.PivotPoint, self.NewMousePos]})
            batch.draw(shader)

            
            batch = batch_for_shader(shader, 'LINES', {"pos": [self.PivotPoint, self.TransfromOrientationMatrix.col[2]*2+self.PivotPoint]})
            batch.draw(shader)
        else:
            super().DrawCallBackBatch()

    def DrawCallBack2D(self, context):
        # Draw Text
        if self.NewMousePos.length != 0:
            self.ShaderUtility.DrawTextInderMouse(self.NewMousePos, self.RotationValue)

    def AfterLeftMouseAction(self, event):
        if event.value == 'PRESS':
            self.ActionsState.MoveMouse = True

            self.StartDrawVector = self.GML(event)
            self.NewMousePos = self.StartDrawVector.copy()
            self.LastAngle = self.GetDirection(self.NewMousePos)

            return self.ORI.RUNNING_MODAL
        elif event.value == 'RELEASE':
            return self.ORI.FINISHED

    def AfterRightMouseAction(self, event):
        axis = GetIndexOfMaxValueInVector(self.NormalIntersectionPlane)
        SetConstarin.SetRotationOnly(axis)
        return self.ORI.FINISHED

    def BeforeMoveMouseAction(self, event):
        pass
    
    def AfterMoveMouseAction(self, event):
        self.NewMousePos = self.GML(event)
        angle = self.LastAngle.angle(self.GetDirection(self.NewMousePos))
        angle = math.degrees(angle)
        #deb("","", self.PivotPoint, self.NormalIntersectionPlane)
        self.Rotatate(angle)
        return self.ORI.RUNNING_MODAL

    def Rotatate(self, angle):
        # Check rotation step
        if angle != None and (round(angle / self.AngleSnappingStep) * self.AngleSnappingStep) != 0:
            angle = self.AngleSnappingStep

            # find third axis 1 is mouse direction 2 is view direction  and 3 (corss) look at pivot point
            cross=((self.GetDirection(self.NewMousePos) - self.GetDirection(self.OldMousePos))).normalized().cross(self.NormalIntersectionPlane)

            # if value biger then 0 counterclock-wise else clockwise
            pos_neg = cross.dot(self.GetDirection(self.NewMousePos)) > 0
            angle = angle*-1 if pos_neg > 0 else angle


            self.RotationValue += angle
            # Rotate self.OldMousePos to current rotation
            self.OldMousePos = self.NewMousePos.copy()
            self.LastAngle = self.LastAngle @ mathutils.Matrix.Rotation(math.radians(angle), 3, self.NormalIntersectionPlane)

            SetConstarin.SetRotationOnlyAT(angle , GetIndexOfMaxValueInVector(self.NormalIntersectionPlane))

class AdvancedTransformUV(Operator):
    def __init__(self):
        super().__init__()
        self.PivotPoint = bpy.context.space_data.pivot_point
        self.SnapUVElement = bpy.context.scene.tool_settings.snap_uv_element

        self.Event = bpy.types.Event

        self.ORI = ORI()
        self.ActionsState = ActionsState()
        self.GenerateLambdaConditions()
        self.GenerateDelegates()

    def GenerateLambdaConditions(self):
        """Conditions for action, can be overridden at __init__ at children classes"""
        self.If_Modify = lambda event: event.shift or event.alt
        self.If_Pass = lambda event: (self.If_MMove(event) and self.ActionsState.Pass) and self.SkipFrameCurrent != 0

        self.If_LM = lambda event: event.type == 'LEFTMOUSE'
        self.If_LM_Cond = lambda event: (self.If_LM(event) or self.ActionsState.LeftMouse) and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_MMove = lambda event: event.type == 'MOUSEMOVE' 
        self.If_MMove_Cond = lambda event: False

        self.If_MM = lambda event: event.type == 'MIDDLEMOUSE'
        self.If_MM_Cond = lambda event: self.If_MM(event) or self.ActionsState.MiddleMouse and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_RM = lambda event: event.type == 'RIGHTMOUSE'
        self.If_RM_Cond = lambda event: self.If_RM(event) or self.ActionsState.RightMouse and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_Spcae = lambda event: event.type == 'SPACE'
        self.If_Spcae_Cond = lambda event: self.If_Spcae(event) or self.ActionsState.Space

        self.If_Shift = lambda event: event.shift
        self.If_Shift_Cond = lambda event: self.If_Shift(event) and self.If_MMove(event)

        self.If_Alt = lambda event: event.alt
        self.If_Alt_Cond = lambda event: self.If_Alt(event) and self.If_MMove(event)

        self.If_Ctrl = lambda event: event.ctrl
        self.If_Ctrl_Cond = lambda event: self.If_Ctrl(event) or self.ActionsState.Ctrl

        self.If_Esc = lambda event: event.type == 'ESC'
        self.If_Esc_Cond = lambda event: self.If_Esc(event) or self.ActionsState.Esc

        self.If_G = lambda event: event.unicode == 'G' or event.unicode == 'g'
        self.If_X = lambda event: event.unicode == 'X' or event.unicode == 'x'
        self.If_Y = lambda event: event.unicode == 'Y' or event.unicode == 'y'
        self.If_Z = lambda event: event.unicode == 'Z' or event.unicode == 'z'

    def GenerateDelegates(self):
        self.LM_D = lambda event: self.StartDelay(event, self.AfterLeftMouseAction, self.BedoreLeftMouseAction, use_delay=True)
        self.RM_D = lambda event: self.StartDelay(event, self.AfterRightMouseAction, self.BeforeRightMouseAction)
        self.MM_D = lambda event: self.StartDelay(event, self.AfterMiddleMouseAction, self.BeforeMiddleMouseAction)
        self.MoveM_D = lambda event: self.StartDelay(event, self.AfterMoveMouseAction, self.BeforeMoveMouseAction)
        self.Space_D = lambda event: self.StartDelay(event, self.AfterSpaceAction, self.BeforeSpaceAction)
        self.Shift_D = lambda event: self.StartDelay(event, self.AfterShiftAction , self.BeforeShiftAction)
        self.Alt_D = lambda event : self.StartDelay(event, self.AfterAltAction , self.BeforeAltAction)
        self.Ctrl_D = lambda event : self.StartDelay(event, self.AfterCtrlAction , self.BeforeCtrlAction)

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

    def Exit(self):
        try:
            if self._handle_3d: bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, 'WINDOW')
            if self._handle_2d: bpy.types.SpaceView3D.draw_handler_remove(self._handle_2d, 'WINDOW')
        except:
            pass
        self.UserSettings.ReturnAllSettings()
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
        return context.space_data.type == "IMAGE_EDITOR"
    
    def modal(self, context, event):
        pass

    def modal(self, context, event):
        if self.If_Esc(event): return self.ModalReturnDec(self.Canceled())
        # -----------------------Skip Frames-------------------------------------------------------------#
        if self.If_Pass(event):
            self.SkipFrameCurrent -= 1
            return self.ORI.RUNNING_MODAL
        elif self.Expected_Action and self.ActionsState.Pass and self.SkipFrameCurrent <= 0: 
            self.ActionsState.Pass = False
            self.NewMousePos = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
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

        if self.If_G(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.G, self.After_G, self.Before_G))
        if self.If_X(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.X, self.After_X, self.Before_X))
        if self.If_Y(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.Y, self.After_Y, self.Before_Y))
        if self.If_Z(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.Z, self.After_Z, self.Before_Z))
        
        return self.ORI.RUNNING_MODAL

        if context.space_data.type == 'IMAGE_EDITOR':
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a IMAGE_EDITOR")
            return {'CANCELLED'}

class AdvancedMoveUV(Operator):
    ''' Advanced move '''
    bl_idname = "view3d.advancedmove_uv"
    bl_label = "Advanced Move UV"



    def modal(self, context, event):

        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                cursor = area.spaces.active.cursor_location
                # print (cursor)
        # -----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------

        if event.type == 'LEFTMOUSE' or self.LB:
            if self.temp_loc_first is None:
                self.temp_loc_first = event.mouse_region_x, event.mouse_region_y
            self.LB = True
            if event.value == 'PRESS':
                if self.count_step <= 2:
                    self.count_step += 1
                    return {'RUNNING_MODAL'}
                else:
                    self.temp_loc_last = event.mouse_region_x, event.mouse_region_y
                    self.axis = SetupAxisUV(self)
                    SetConstarin.SetMoveOnlyUV(self, self.axis)
                return {'RUNNING_MODAL'}
            elif not event.is_repeat:
                UserPresets(self, True)
                return {'FINISHED'}

            # -----------------------RIGHT_MOUSE Exlude Axis------------------------------------------------------

        elif event.type == 'RIGHTMOUSE' or self.RB:
            if event.value == 'PRESS':
                self.RB = True
                SetConstarin.SetMoveExcludeUV(self)
            elif not event.is_repeat:
                UserPresets(self, True)
                return {'FINISHED'}
        #
        #     # -----------------------MIDDLE_MOUSE No Constrain---------------------------------------------------
        #
        # elif event.type == 'MIDDLEMOUSE' or self.MB:
        #     self.MB = True
        #     if event.value == 'PRESS':
        #         SetConstarin.SetMoveNoConstrainNoSnap(self, context)
        #     elif not event.is_repeat:
        #         UserPresets(self, context, False)
        #         return {'FINISHED'}
        #
        #     # -----------------------SPACEBAR Bottom            --------------------------------------------------
        # elif event.type == 'SPACE' or self.SPACE:
        #     self.SPACE = True
        #
        #     if event.value == 'PRESS':
        #         SnapMoveOrientation(self, context)
        #         SetConstarin.SetMoveNoConstrain(self, context)
        #         return {'RUNNING_MODAL'}
        #
        # elif event.unicode == 'G' or event.unicode == 'g':
        #     if context.mode == "EDIT_MESH":
        #         bpy.ops.transform.edge_slide('INVOKE_DEFAULT')
        #         UserPresets(self, True)
        #         return {'FINISHED'}
        # elif event.unicode == 'X' or event.unicode == 'x':
        #     bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
        #     UserPresets(self, True)
        #     return {'FINISHED'}
        # elif event.unicode == 'Y' or event.unicode == 'y':
        #     bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
        #     UserPresets(self, True)
        #     return {'FINISHED'}
        # elif event.unicode == 'Z' or event.unicode == 'z':
        #     bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
        #     UserPresets(self, True)
        #     return {'FINISHED'}

        if event.type == 'ESC':
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        # context.area.header_text_set(
        # 'Drag LMB constraint axis,
        # RMB translate by two axis,
        # MMB free translate,
        # SPACE free translate with snap and rotate along normal'
        # )
        if context.space_data.type == 'IMAGE_EDITOR':
            # UserSnap(self, context)
            UserPresets(self, context)
            # GetDirection(self)
            # GetCenter(self)
            # GetView(self)
            #
            self.LB = False
            self.RB = False
            # self.MB = False
            # self.SPACE = False
            self.temp_loc_first = None
            self.temp_loc_last = None
            self.count_step = 0

            # args = (self, context)
            # self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_VIEW')
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

class AdvancedScaleUV(Operator):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale_uv"
    bl_label = "Advanced Scale UV"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "IMAGE_EDITOR"

    def modal(self, context, event):
        # -----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------------

        if event.type == 'LEFTMOUSE' or self.LB:
            if self.temp_loc_first is None:
                self.temp_loc_first = event.mouse_region_x, event.mouse_region_y
            self.LB = True
            if event.value == 'PRESS':
                if self.count_step <= 2:
                    self.count_step += 1
                    return {'RUNNING_MODAL'}
                else:
                    self.temp_loc_last = event.mouse_region_x, event.mouse_region_y
                    self.axis = SetupAxisUV(self)
                    SetConstarin.SetScaleOnly(self, context, self.axis)
                return {'RUNNING_MODAL'}
            elif not event.is_repeat:
                UserPresets(self, True)
                return {'FINISHED'}

            # -----------------------RIGHT_MOUSE Exlude Axis---------------------------------------------------------

        elif event.type == 'RIGHTMOUSE' or self.RB:
            if event.value == 'PRESS':
                self.RB = True
                SetConstarin.SetScaleExcludeUV(self, context, self.exc_axis)
            elif not event.is_repeat:
                UserPresets(self, True)
                return {'FINISHED'}

            # -----------------------SPACEBAR Bottom Setup zero-------------------------------------------------------

        elif event.type == 'SPACE' or self.SPACE:
            if self.temp_loc_first is None:
                self.temp_loc_first = event.mouse_region_x, event.mouse_region_y
            self.SPACE = True
            if event.value == 'PRESS':
                if self.count_step <= 2:
                    self.count_step += 1
                    return {'RUNNING_MODAL'}
                else:
                    self.temp_loc_last = event.mouse_region_x, event.mouse_region_y
                    self.axis = SetupAxisUV(self)
                    SetConstarin.SetScaleOnlySetZero(self, context, self.axis)
                    UserPresets(self, True)
                    return {'FINISHED'}

                # -----------------------ALT for negative value-----------------------------------------------------

        # elif event.type == 'BUTTON4MOUSE' or self.ALT:
        elif event.shift or self.ALT:
            if self.temp_loc_first is None:
                self.temp_loc_first = event.mouse_region_x, event.mouse_region_y
            self.ALT = True
            if event.value == 'PRESS':
                if self.count_step <= 2:
                    self.count_step += 1
                    return {'RUNNING_MODAL'}
                else:
                    if context.mode == "EDIT_MESH":
                        bpy.ops.mesh.flip_normals()
                    self.temp_loc_last = event.mouse_region_x, event.mouse_region_y
                    self.axis = SetupAxisUV(self)
                    SetConstarin.SetScaleMirror(self, context, self.axis)
                    UserPresets(self, True)
                    return {'FINISHED'}

        elif event.unicode == 'X' or event.unicode == 'x':
            bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, False))
            UserPresets(self, True)
            return {'FINISHED'}
        elif event.unicode == 'Y' or event.unicode == 'y':
            bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, False))
            UserPresets(self, True)
            return {'FINISHED'}

        if event.type == 'ESC':
            UserPresets(self, True)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        # context.area.header_text_set(
        # 'Drag LMB constraint axis,
        # RMB resize by two axis,
        # MMB free resize,
        # SHIFT mirror,
        # SPACE flatten'
        # )
        if context.space_data.type == 'IMAGE_EDITOR':
            # UserSnap(self, context)
            UserPresets(self, context)
            # GetDirection(self)
            # GetCenter(self)
            # GetView(self)
            self.LB = False
            self.RB = False
            self.MB = False
            self.SPACE = False
            self.ALT = False
            self.temp_loc_first = None
            self.temp_loc_last = None
            self.count_step = 0

            # args = (self, context)
            # self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_VIEW')
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

class AdvancedRotationUV(Operator):
    ''' Advanced move '''
    bl_idname = "view3d.advancedrotation_uv"
    bl_label = "Advanced Rotation UV"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "IMAGE_EDITOR"

    def modal(self, context, event):
        self.temp_loc_last = event.mouse_region_x, event.mouse_region_y

        if event.type == 'LEFTMOUSE' or self.LB:
            self.temp_loc_last = event.mouse_region_x, event.mouse_region_y
            if event.value == 'PRESS':
                self.LB = True
                if self.LB_cal:
                    self.LB_cal = False
                    self.exc_axis = 'z'
                    SetConstarin.SetRotationOnly(self, context, self.exc_axis)
                return {'RUNNING_MODAL'}
            elif not event.is_repeat:
                UserPresets(self, True)
                return {'FINISHED'}

        elif event.type == 'RIGHTMOUSE' or self.RB:
            self.temp_loc_last = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
            if self.temp_loc_first is None:
                self.temp_loc_first = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 

            if event.value == 'PRESS':
                self.RB = True
                if int(round(self.temp_agle)) in range(0, 24) and (int(round(self.temp_agle)) != int(self.temp)):
                    self.temp = int(round(self.temp_agle))

                    if self.exc_axis == 'y':
                        self.agle = self.temp * -1
                    else:
                        self.agle = self.temp

                    if self.pre_rot != 0:
                        # if self.pre_rot >self.agle*15:
                        #     self.agle = (self.pre_rot)-(self.agle*15)
                        # else:
                        #     self.agle = (self.agle * 15)- (self.pre_rot)
                        SetConstarin.SnapRotation(self, context, self.pre_rot * -1, self.exc_axis)
                    # else:
                    # self.agle*=15
                    self.pre_rot = self.agle * 15
                    SetConstarin.SnapRotation(self, context, self.agle * 15, self.exc_axis)

                return {'RUNNING_MODAL'}
            elif not event.is_repeat:
                UserPresets(self, context, False)
                SetUserSnap(self, context)
                DeleteOrientation(self, context)
                context.window.scene.transform_orientation_slots[0].type = user_orient
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                except:
                    pass
                return {'FINISHED'}

        if event.type == 'ESC':
            UserPresets(self, True)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        # context.area.header_text_set(
        # 'LMB constraint view axis,
        # RMB constraint view axis snap 15 degrees,
        # MMB free rotate')
        # self.time = time.clock()
        if context.space_data.type == 'IMAGE_EDITOR':
            UserPresets(self, context)

            self.LB = False
            self.LB_cal = True
            self.RB = False
            self.MB = False
            self.SPACE = False
            self.temp_loc_first = None
            self.temp_loc_last = None
            self.count_step = 0
            # args = (self, context)
            # self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_rot, args, 'WINDOW', 'POST_PIXEL')
            self.temp_loc_first = event.mouse_region_x, event.mouse_region_y
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

addon_keymaps = []

def get_addon_preferences():
    ''' quick wrapper for referencing addon preferences '''
    addon_preferences = bpy.context.user_preferences.addons[__name__].preferences
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
    # kmi.properties.name = "view3d.advancedmove"
    kmi.active = True
    addon_keymaps.append((km, kmi))

    wm1 = bpy.context.window_manager
    kc1 = wm1.keyconfigs.addon
    km1 = kc1.keymaps.new(name="3D View Generic", space_type='VIEW_3D', region_type='WINDOW')
    kmi1 = km1.keymap_items.new(AdvancedScale.bl_idname, 'S', 'PRESS', shift=False, ctrl=False, alt=False)
    # kmi.properties.name = "view3d.advancedmove"
    kmi1.active = True
    addon_keymaps.append((km1, kmi1))

    wm2 = bpy.context.window_manager
    kc2 = wm2.keyconfigs.addon
    km2 = kc2.keymaps.new(name="3D View Generic", space_type='VIEW_3D', region_type='WINDOW')
    kmi2 = km2.keymap_items.new(AdvancedRotation.bl_idname, 'R', 'PRESS', shift=False, ctrl=False, alt=False)
    # kmi.properties.name = "view3d.advancedmove"
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
    )
    Use_Advanced_Transform: BoolProperty(
        name="Use Advanced Transform Rotation",
        default=False,
        description="Is not use standard blender rotation without snapping (left mouse button)."
                    "Standard has great performance.",
    )

    Draw_Tooll: BoolProperty(
        name="Draw Tool",
        default=True,
        description="Draw Active Tool.",
    )

    def draw(self, context):
        layout = self.layout
        box0 = layout.box()
        row1 = box0.row()
        row2 = box0.row()
        row3 = box0.row()
        row1.prop(self, "Snapping_Step")
        row2.prop(self, "Use_Advanced_Transform")
        row3.prop(self, "Draw_Tooll")
        # ---------------------------------
        box = layout.box()
        split = box.split()
        col = split.column()
        # col.label("Setup Advanced Move")
        col.separator()
        wm = bpy.context.window_manager
        kc = wm.keyconfigs.user
        km = kc.keymaps['3D View Generic']
        kmi = get_hotkey_entry_item(km, "view3d.advancedmove")
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
        # col.label("Setup Advanced Move")
        col1.separator()
        wm1 = bpy.context.window_manager
        kc1 = wm1.keyconfigs.user
        km1 = kc1.keymaps['3D View Generic']
        kmi1 = get_hotkey_entry_item(km1, "view3d.advancedscale")
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
        # col.label("Setup Advanced Move")
        col2.separator()
        wm2 = bpy.context.window_manager
        kc2 = wm2.keyconfigs.user
        km2 = kc2.keymaps['3D View Generic']
        kmi2 = get_hotkey_entry_item(km2, "view3d.advanced_rotation")
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
    def SetScaleOnly(axis):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetScaleExclude(axis):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=SetConstarin.SetupExcludeAxis(axis))
    @staticmethod
    def SetScaleNoConstrain():
        bpy.ops.transform.resize('INVOKE_DEFAULT')
    @staticmethod
    def SetScaleOnlySetZero(axis):
        axis_value = {0: (0.0, 1.0, 1.0), 1: (1.0, 0.0, 1.0), 2: (1.0, 1.0, 0.0)}
        bpy.ops.transform.resize(value=axis_value[axis], constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetScaleExcludeSetZero(axis):
        axis_value = {0: (0.0, 1.0, 1.0), 1: (1.0, 0.0, 1.0), 2: (1.0, 1.0, 0.0)}
        bpy.ops.transform.resize(value=axis_value[axis], constraint_axis=SetConstarin.SetupExcludeAxis[axis])
    @staticmethod
    def SetScaleMirror(axis):
        axis_value = {0: (-1.0, 1.0, 1.0), 1: (1.0, -1.0, 1.0), 2: (1.0, 1.0, -1.0)}
        print(axis,"adfasdf")
        bpy.ops.transform.resize(value=axis_value[axis], constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SnapRotation(axis, value):
        bpy.ops.transform.rotate(value=math.radians(value),constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetMoveOnlyUV(axis):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetMoveExcludeUV():
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, True, False))


classes = (AdvancedTransform, AdvancedMove, AdvancedScale, AdvancedRotation, AdvancedMoveUV, AdvancedScaleUV, AdvancedRotationUV,
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
