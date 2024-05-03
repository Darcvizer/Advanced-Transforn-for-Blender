import math
import os

import copy
import blf
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
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix
from mathutils import Vector, Matrix
from mathutils import Vector as V
from mathutils.geometry import intersect_line_plane, intersect_point_quad_2d, intersect_line_line_2d

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

DubegMode = True
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
        null.location = pos if pos != None else Vector((0.0,0.0,0.0))
        null.matrix_world = dir.to_track_quat('Z', 'Y').to_matrix().to_4x4() if dir != None else Vector((0.0,0.0,1.0)).to_track_quat('Z', 'Y').to_matrix().to_4x4()
        null.name = "ADHelpNull"
        bpy.ops.object.constraint_add(type='COPY_TRANSFORMS')
        bpy.context.object.constraints["Copy Transforms"].target = plane

    if DubegMode:
       
        print(dis if dis != None and isinstance(dis, str) else "",': ' , value,)
        if isinstance(dis, Vector):
            pos = dis
        if pos != None and pos.length == 1:
            dir = pos
        plane = bpy.data.objects.get("ADHelpPlane")
        null = bpy.data.objects.get("ADHelpNull")
        if  pos != None or dir != None:
            if (plane and null):
                if plane and pos != None:
                    plane.location = pos
                if plane and (dir != None and dir):
                    plane.matrix_world = dir.to_track_quat('Z', 'Y').to_matrix().to_4x4()

                if null and pos != None:
                    plane.location = pos
                if null and (dir != None and dir):
                    null.matrix_world = dir.to_track_quat('Z', 'Y').to_matrix().to_4x4()
            else:
                CreateObjects()
        if forced:
            CreateObjects()
                

def GetNormalForIntersectionPlane(TransfromOrientationMatrix):
    view_direction = bpy.context.region_data.view_rotation @ Vector((0, 0, -1))
    index_best_axis = [abs(view_direction.dot(TransfromOrientationMatrix.col[i])) for i in range(0,3)] # Get dot for all axes
    index_best_axis = index_best_axis.index(max(index_best_axis))
    normal_intersect_plane = (TransfromOrientationMatrix.col[index_best_axis] * -1).normalized()# do invert for normal because we need derection to camera
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
    # Create new orientation it is the easiest way to get matrix for current orientation. For example for geting normal we have to create bimesh, bad way
    temp = bpy.context.scene.transform_orientation_slots[0].type
    bpy.ops.transform.create_orientation(name="AdvancedTransform", use_view=False, use=True,overwrite=True)
    matrix_transform_orientation = bpy.context.scene.transform_orientation_slots[0].custom_orientation.matrix
    bpy.ops.transform.create_orientation()
    bpy.context.scene.transform_orientation_slots[0].type = temp
    return matrix_transform_orientation

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

def GetAxixForTransformation(normal):
    """Get axis by mouse direaction
    normal: vector"""
    index_of_max_value = max(range(len(normal)), key=lambda i: abs(normal[i])) # get index of maximal value at vector
    return ['x','y','z'][index_of_max_value] # create array and get value by index

def GetMouseDirection(point_from, point_to, matrix):
    """Get mouse direction by 2 saved points, also if we have specific transform orientation we have to rotate vector by orientation matrix
    point_from: Vector | mouse point 1
    point_to: Vector | mouse point 2
    matrix: Matrix | Matrix for rotation vector
    """
    direction = (matrix.to_3x3().inverted() @ (point_from - point_to).normalized()) * -1 # Rotate move direction to transform orientation matrix
    return GetAxixForTransformation(direction)

def UserPresets(self, Set=False):
    if Set is False:
        self.save_user_drag = context.preferences.inputs.use_drag_immediately
        if self.save_user_drag is not True:
            context.preferences.inputs.use_drag_immediately = True

    elif Set is True:
        context.preferences.inputs.use_drag_immediately = self.save_user_drag

def UserSnap(self, Set=False):
    context = bpy.context
    if Set == False:
        self.user_snap_element = context.scene.tool_settings.snap_elements
        self.user_snap_target = context.scene.tool_settings.snap_target
        self.user_snap_rot = context.scene.tool_settings.use_snap_align_rotation
        # self.user_snap_project = context.scene.tool_settings.use_snap_project
        self.user_snap_project = context.scene.tool_settings.snap_elements_individual
        self.user_snap = context.scene.tool_settings.use_snap
    elif Set:
        context.scene.tool_settings.snap_elements = self.user_snap_element
        context.scene.tool_settings.snap_target = self.user_snap_target
        context.scene.tool_settings.use_snap_align_rotation = self.user_snap_rot
        # context.scene.tool_settings.use_snap_project = self.user_snap_project
        context.scene.tool_settings.snap_elements_individual = self.user_snap_project
        context.scene.tool_settings.use_snap = self.user_snap

def CalculateCirclePoint(self):
    self.MailCircle_3d = []
    self.SecondPoint_3d = []
    self.MailCircle_2d = []
    self.SecondPoint_2d = []

    self.ShortLine = []
    self.LongLine = []
    self.LongLine_2d = []
    self.SnapLine = []
    self.SnapLine_2d = []
    self.edging1 = []
    self.edging2 = []
    self.edging3 = []

    self.MainCirclePositiv = []
    self.MainCircleNegativ = []
    self.SecondCirclePositiv = []
    self.SecondCircleNegativ = []
    self.firstPoint = None
    self.secondPoint = None
    self.color = V((0.8, 0.6, 0.0, 0.2))

    self.line = []

    def c(v, f):
        v.normalized()
        return v * f + v

    # ------------------------Size Gizmo---------------------#
    for i in bpy.context.window.screen.areas:
        if i.type == 'VIEW_3D':
            CD = i.spaces[0].region_3d.view_distance / 20
            if CD >= 0.3:
                MS = Matrix.Scale(CD, 4, )
            else:
                MS = Matrix.Scale(0.3, 4, )
    step = 0
    # -----Get full circle 3d-----#
    for i in range(1, 361):
        step += 1

        angle = 2 * math.pi * i / 360
        x = 5 * math.cos(angle)
        y = 5 * math.sin(angle)
        self.MailCircle_3d.append(V((x, y, 0)))
        self.MainCirclePositiv.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                            (self.RotaionMatrix @ MS) @ c(
                                                                                self.MailCircle_3d[-1], -0.1)))
        self.MainCircleNegativ.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                            (self.NegativeMatrix @ MS) @ c(
                                                                                self.MailCircle_3d[-1], -0.1)))
        angle = 2 * math.pi * (i - 1) / 360
        x = 5 * math.cos(angle)
        y = 5 * math.sin(angle)
        self.MailCircle_3d.append(V((x, y, 0)))
        self.MainCirclePositiv.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                            (self.RotaionMatrix @ MS) @ c(
                                                                                self.MailCircle_3d[-1], -0.1)))
        self.MainCircleNegativ.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                            (self.NegativeMatrix @ MS) @ c(
                                                                                self.MailCircle_3d[-1], -0.1)))

        self.MainCirclePositiv.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                            (self.RotaionMatrix @ MS) @ V((0, 0, 0))))
        self.MainCircleNegativ.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                            (self.NegativeMatrix @ MS) @ V((0, 0, 0))))

    offset = 0.9
    step2 = 0
    for i in range(0, len(self.MailCircle_3d) - 2):
        self.SecondCirclePositiv.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.RotaionMatrix @ MS) @ self.MailCircle_3d[i]))
        self.SecondCirclePositiv.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.RotaionMatrix @ MS) @ self.MailCircle_3d[i + 1]))
        self.SecondCirclePositiv.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.RotaionMatrix @ MS) @ V((self.MailCircle_3d[i][0] * offset,
                                                                                 self.MailCircle_3d[i][1] * offset,
                                                                                 self.MailCircle_3d[i][
                                                                                     2] * offset))))  # + coords[i]
        self.SecondCirclePositiv.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.RotaionMatrix @ MS) @ V((self.MailCircle_3d[i][0] * offset,
                                                                                 self.MailCircle_3d[i][1] * offset,
                                                                                 self.MailCircle_3d[i][
                                                                                     2] * offset))))  # + coords[i]
        self.SecondCirclePositiv.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.RotaionMatrix @ MS) @ V((self.MailCircle_3d[i + 1][0] * offset,
                                                                                 self.MailCircle_3d[i + 1][1] * offset,
                                                                                 self.MailCircle_3d[i + 1][
                                                                                     2] * offset))))  # + coords[i+1]
        self.SecondCirclePositiv.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.RotaionMatrix @ MS) @ self.MailCircle_3d[i + 1]))

        self.SecondCircleNegativ.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.NegativeMatrix @ MS) @ (self.MailCircle_3d[i])))
        self.SecondCircleNegativ.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.NegativeMatrix @ MS) @ (self.MailCircle_3d[i + 1])))
        self.SecondCircleNegativ.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.NegativeMatrix @ MS) @ (V((self.MailCircle_3d[i][0] * offset,
                                                                                   self.MailCircle_3d[i][1] * offset,
                                                                                   self.MailCircle_3d[i][
                                                                                       2] * offset)))))  # + coords[i]
        self.SecondCircleNegativ.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.NegativeMatrix @ MS) @ (V((self.MailCircle_3d[i][0] * offset,
                                                                                   self.MailCircle_3d[i][1] * offset,
                                                                                   self.MailCircle_3d[i][
                                                                                       2] * offset)))))  # + coords[i]
        self.SecondCircleNegativ.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.NegativeMatrix @ MS) @ (V((
                                                      self.MailCircle_3d[i + 1][0] * offset,
                                                      self.MailCircle_3d[i + 1][1] * offset,
                                                      self.MailCircle_3d[i + 1][
                                                          2] * offset)))))  # + coords[i+1]
        self.SecondCircleNegativ.append(
            view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                  (self.NegativeMatrix @ MS) @ (self.MailCircle_3d[i + 1])))

        self.edging1.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                  (self.RotaionMatrix @ MS) @ self.MailCircle_3d[i]))
        self.edging2.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                  (self.RotaionMatrix @ MS) @ (
                                                                      c(self.MailCircle_3d[i], -0.1))))

    self.edging1.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                              (self.RotaionMatrix @ MS) @ self.MailCircle_3d[
                                                                  len(self.MailCircle_3d) - 1]))
    self.edging2.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                              (self.RotaionMatrix @ MS) @ (
                                                                  c(self.MailCircle_3d[len(self.MailCircle_3d) - 1],
                                                                    -0.1))))

    # -----Get line
    flip_flop = True
    offsetLine = 0.1
    offset = 0.5
    step = 0
    for i in range(0, len(self.MailCircle_3d) - 1, 20):  # 0# 20 #+1
        self.LongLine_2d.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (self.RotaionMatrix @ MS) @ (
                                                                          c(self.MailCircle_3d[i + 1], -0.025))))
        self.LongLine_2d.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (self.RotaionMatrix @ MS) @ (
                                                                          c(self.MailCircle_3d[i + 1], -0.075))))

        self.LongLine_2d.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (self.RotaionMatrix @ MS) @ (
                                                                          c(self.MailCircle_3d[i + 1 + 10], 0))))
        self.LongLine_2d.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (self.RotaionMatrix @ MS) @ (
                                                                          c(self.MailCircle_3d[i + 1 + 10], -0.1))))

    self.firstPoint = (view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                             (self.RotaionMatrix @ MS) @ V((0, 0, 0))))

def f(self, context, event):
    # print("self.Angle", self.Angle)
    if self.call_no_snap == 20:
        self.mousePosition = event.mouse_region_x, event.mouse_region_y
        self.temp_loc_last = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
        self.Angle = int(round(
            GetAngle(self, self.temp_loc_last, self.temp_loc_first, self.V_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)))))
        self.realAngle = self.Angle
        intersect_Quad(self, event)
        self.call_no_snap = 0
    else:
        self.call_no_snap += 1

def DrawCalback(self, context, event):
    try:

        # ------------------Draw 2D Quad-----------------
        # coord = [self.LD, self.CD, self.C, self.LC]
        #
        # indices = ((0, 1, 2), (2, 3, 0))
        # batch = batch_for_shader(shader, 'TRIS', {"pos": coord}, indices=indices)
        # shader.bind()
        # shader.uniform_float("color", (1.0, 0.0, 0.0, 0.25))
        # batch.draw(shader)
        #
        # coord = [self.CD, self.RD, self.RC, self.C]
        # indices = ((0, 1, 2), (2, 3, 0))
        # batch = batch_for_shader(shader, 'TRIS', {"pos": coord}, indices=indices)
        #
        # shader.bind()
        # shader.uniform_float("color", (0.0, 1.0, 0.0, 0.25))
        # batch.draw(shader)
        #
        # coord = [self.C, self.RC, self.RU, self.CU]
        # indices = ((0, 1, 2), (2, 3, 0))
        # batch = batch_for_shader(shader, 'TRIS', {"pos": coord}, indices=indices)
        #
        # shader.bind()
        # shader.uniform_float("color", (0.0, 0.0, 1.0, 0.25))
        # batch.draw(shader)
        #
        # coord = [self.LC, self.C, self.CU, self.LU]
        # indices = ((0, 1, 2), (2, 3, 0))
        # batch = batch_for_shader(shader, 'TRIS', {"pos": coord}, indices=indices)
        #
        # shader.bind()
        # shader.uniform_float("color", (1.0, 1.0, 1.0, 0.25))
        # batch.draw(shader)
        if not self.is_snap:
            f(self, context, event)

        shader = ggpu.shader.from_builtin('UNIFORM_COLOR')
        # gpu.state.line_width_set(3)
        # gpu.state.blend_set("ALPHA")

        # ----------------Draw main circle---------------------
        if self.ModeRotation:
            batch = batch_for_shader(shader, 'TRIS', {"pos": self.MainCircleNegativ[:(self.realAngle * 3) * -1]})
            shader.uniform_float("color", self.color)
            batch.draw(shader)
        else:
            batch = batch_for_shader(shader, 'TRIS', {"pos": self.MainCirclePositiv[:self.realAngle * 3]})
            shader.uniform_float("color", self.color)
            batch.draw(shader)

        if self.ModeRotation:
            batch = batch_for_shader(shader, 'TRIS', {"pos": self.SecondCircleNegativ[(self.Angle * 12) * -1:]})
            shader.uniform_float("color", (0.5, 0.5, 0.5, 0.1))
            batch.draw(shader)
        else:
            batch = batch_for_shader(shader, 'TRIS', {"pos": self.SecondCirclePositiv[self.Angle * 12:]})
            shader.uniform_float("color", (0.5, 0.5, 0.5, 0.1))
            batch.draw(shader)

        if self.ModeRotation:
            batch = batch_for_shader(shader, 'TRIS', {"pos": self.SecondCircleNegativ[:(self.Angle * 12) * -1]})
            shader.uniform_float("color", (1, 0.4, 0.2, 0.2))
            batch.draw(shader)
        else:
            batch = batch_for_shader(shader, 'TRIS', {"pos": self.SecondCirclePositiv[:self.Angle * 12]})
            shader.uniform_float("color", (1, 0.4, 0.2, 0.2))
            batch.draw(shader)

        gpu.state.line_width_set(2)

        batch = batch_for_shader(shader, 'LINES', {"pos": self.LongLine_2d})
        shader.uniform_float("color", (0, 0, 0, 0.5))
        batch.draw(shader)

        gpu.state.line_width_set(3)

        batch = batch_for_shader(shader, 'LINES', {"pos": self.SnapLine_2d})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.05))
        batch.draw(shader)

        batch = batch_for_shader(shader, 'LINES', {"pos": self.edging1})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
        batch.draw(shader)

        batch = batch_for_shader(shader, 'LINES', {"pos": self.edging2})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
        batch.draw(shader)

        line = []
        line.append(self.firstPoint)
        line.append(self.edging2[0])
        line.append(self.edging2[self.realAngle * 2])
        line.append(self.firstPoint)

        batch = batch_for_shader(shader, 'LINES', {"pos": line})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
        batch.draw(shader)

        blf.position(0, self.mousePosition[0], self.mousePosition[1], 0)
        blf.size(0, 22, 40)
        blf.draw(0, str(round(self.Angle)))

    except:
        pass

def GetAngle(self, v2, v1, n, radian=False):
    v1 = v1 - self.center
    v2 = v2 - self.center

    def normalize(v):
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            norm = np.finfo(v.dtype).eps
        return v / norm

    n = normalize(n)

    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    det = v1.x * v2.y * n.z + v2.x * n.y * v1.z + n.x * v1.y * v2.z - v1.z * v2.y * n.x - v2.z * n.y * v1.x - n.z * v1.y * v2.x
    if radian:
        return math.atan2(det, dot)
    else:
        if self.addAngle != 0:
            return math.degrees(math.atan2(det * -1, dot * -1)) + self.addAngle
        elif self.addAngle == 0:
            return math.degrees(math.atan2(det, dot))  # + self.addAngle

def StartDraw(self, context, event):
    '''Setup setings for rotation'''
    # -----------------Rotation Matrix on directio the first click------------------#
    radians = GetAngle(self, (self.V_matrix @ Vector((0.0, 1.0, 0.0))), self.temp_loc_first,
                       self.V_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)), radian=True)
    # print("radians - ", radians)
    axis_dst = Vector((0.0, 1.0, 0.0))
    self.directionMatrix = (self.temp_loc_first).normalized()
    # matrix_rotate = Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0))).to_3x3() @ (axis_dst.rotation_difference(self.V_matrix.to_3x3().inverted() @ ((self.directionMatrix)).normalized()).to_matrix().to_3x3())
    matrix_rotate = Matrix.Rotation(radians + -1.570796, 3, Vector((0.0, 0.0, -1.0))).to_4x4()
    self.RotaionMatrix = Matrix.Translation(self.center) @ (self.V_matrix.to_3x3() @ matrix_rotate.to_3x3()).to_4x4()
    NM = Matrix.Rotation(-1.570796 * 2, 3, Vector((1.0, 0.0, 0.0))).to_3x3()
    self.NegativeMatrix = Matrix.Translation(self.center) @ (self.RotaionMatrix.to_3x3() @ NM.to_3x3()).to_4x4()
    # Matrix.Translation(self.center) @
    # -------------------Get circle size---------------------#
    for i in bpy.context.window.screen.areas:
        if i.type == 'VIEW_3D':
            CD = i.spaces[0].region_3d.view_distance
            MS = Matrix.Scale(1, 4)

    def Rot(self, x, y):
        """Rotation 2d quad on the first click"""
        # v1 = V((C.region.width/2,C.region.height /2 ))
        v1 = view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)
        v2 = view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.temp_loc_first)
        radians = math.atan2(v1[1] - v2[1], v1[0] - v2[0]) * -1

        # print("angle", math.degrees(radians))
        # print("v1",v1)
        # print("v2",v2)
        # print("radians",radians)
        # origin = V((C.region.width/2,C.region.height /2 ))
        origin = view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)
        centerOffset = origin - V((bpy.context.region.width / 2, bpy.context.region.height / 2))
        offset_x = origin[0]
        offset_y = origin[1]
        x = x + centerOffset[0]
        y = y + centerOffset[1]
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return V((qx, qy))

    # Высота C.region.height
    # Ширена C.region.width
    zero = (bpy.context.region.width * 2)
    max = (bpy.context.region.width * 2) * -1

    self.LD = Rot(self, V((zero, zero))[0], V((zero, zero))[1])
    self.LC = Rot(self, V((zero, zero))[0], V((zero, bpy.context.region.height / 2))[1])
    self.LU = Rot(self, V((zero, bpy.context.region.height + max))[0], V((zero, bpy.context.region.height + max))[1])
    self.RD = Rot(self, V((bpy.context.region.width + max, zero))[0], V((bpy.context.region.width + max, zero))[1])
    self.RC = Rot(self, V((bpy.context.region.width + max, bpy.context.region.height / 2))[0],
                  V((bpy.context.region.width + max, bpy.context.region.height / 2))[1])
    self.RU = Rot(self, V((bpy.context.region.width + max, bpy.context.region.height + max))[0],
                  V((bpy.context.region.width + max, bpy.context.region.height + max))[1])
    self.C = Rot(self, V((bpy.context.region.width / 2, bpy.context.region.height / 2))[0],
                 V((bpy.context.region.width / 2, bpy.context.region.height / 2))[1])
    self.CU = Rot(self, V((bpy.context.region.width / 2, bpy.context.region.height + max))[0],
                  V((bpy.context.region.width / 2, bpy.context.region.height + max))[1])
    self.CD = Rot(self, V((bpy.context.region.width / 2, zero))[0], V((bpy.context.region.width / 2, zero))[1])

    # print("#-----------#")
    # print("self.LD", self.LD)
    # print("self.LC", self.LC)
    # print("self.LU", self.LU)
    # print("self.RD", self.RD)
    # print("self.RC", self.RC)
    # print("self.RU", self.RU)
    # print("self.C" , self.C )
    # print("self.CU", self.CU)
    # print("self.CD", self.CD)
    # print("#-----------#")

def intersect_Quad(self, event):
    mouse = event.mouse_region_x, event.mouse_region_y
    red = intersect_point_quad_2d(mouse, self.LD, self.CD, self.C, self.LC)
    green = intersect_point_quad_2d(mouse, self.CD, self.RD, self.RC, self.C)
    blue = intersect_point_quad_2d(mouse, self.C, self.RC, self.RU, self.CU)
    white = intersect_point_quad_2d(mouse, self.LC, self.C, self.CU, self.LU)

    # print("#-----------#")
    # print("red", red)
    # print("green", green)
    # print("blue", blue)
    # print("white", white)
    # print("#-----------#")

    # red = негативный 2 красный
    # green = негативный 1 зеленый
    # blue = позитивный 1 синий
    # white = позитивный 2 белый

    if green:
        if self.activeQuad[0] != "green":
            self.activeQuad[2] = self.activeQuad[1]
            self.activeQuad[1] = self.activeQuad[0]
            self.activeQuad[0] = "green"
            if self.activeQuad[2] == "":
                self.ModeRotation = False
                self.addAngle = 0
            elif self.activeQuad[1] == "green" and self.ModeRotation == True and self.addAngle == 0:
                self.ModeRotation = True
                self.addAngle = 0
            elif self.activeQuad[2] == "white" and self.ModeRotation == True and self.addAngle == 180:
                self.ModeRotation = True
                self.addAngle = 180  # !!!!!!!!!!!
            elif self.activeQuad[1] == "red" and self.ModeRotation == True:
                self.ModeRotation = True
                self.addAngle = 180
            elif self.activeQuad[1] == "red" and self.ModeRotation == False:
                self.ModeRotation = False
                self.addAngle = 0
            elif self.activeQuad[1] == "blue":
                self.ModeRotation = False
                self.addAngle = 0



    elif red:
        if self.activeQuad[0] != "red":
            self.activeQuad[2] = self.activeQuad[1]
            self.activeQuad[1] = self.activeQuad[0]
            self.activeQuad[0] = "red"
            if self.activeQuad[1] == "white" and self.activeQuad[2] == "red" and self.ModeRotation == True:
                self.ModeRotation = True
                self.addAngle = 180  # !!!!!!!!!!!
            elif self.activeQuad[1] == "white" and self.activeQuad[2] == "blue" and self.ModeRotation == True:
                self.ModeRotation = True
                self.addAngle = 180  # !!!!!!!!!!!
            elif self.activeQuad[1] == "green" and self.activeQuad[2] == "red" and self.ModeRotation == True:
                self.ModeRotation = True
                self.addAngle = 180  # !!!!!!!!!!!

            elif self.activeQuad[1] == "white" and self.activeQuad[2] == "":
                self.ModeRotation = True
                self.addAngle = 180  # !!!!!!!!!!!



    elif blue:
        if self.activeQuad[0] != "blue":
            self.activeQuad[2] = self.activeQuad[1]
            self.activeQuad[1] = self.activeQuad[0]
            self.activeQuad[0] = "blue"
            if self.activeQuad[2] == "":
                self.ModeRotation = True
                self.addAngle = 0
            elif self.activeQuad[1] == "green" and self.activeQuad[2] == "red" and self.addAngle == 0:
                self.ModeRotation = True
                self.addAngle = 0
            elif self.activeQuad[1] == "green" and self.ModeRotation == False:
                self.ModeRotation = True
                self.addAngle = 0
            elif self.activeQuad[2] == "red" and self.ModeRotation == False:
                self.ModeRotation = False
                self.addAngle = -180  # !!!!!!!!!!!
            elif self.activeQuad[1] == "white" and self.ModeRotation == True:
                self.addAngle = 0
                self.ModeRotation = True




    elif white:
        if self.activeQuad[0] != "white":
            self.activeQuad[2] = self.activeQuad[1]
            self.activeQuad[1] = self.activeQuad[0]
            self.activeQuad[0] = "white"
            if self.activeQuad[1] == "red" and self.activeQuad[2] == "green" and self.ModeRotation == False:
                self.ModeRotation = False
                self.addAngle = -180  # !!!!!!!!!!!
            elif self.activeQuad[1] == "red" and self.activeQuad[2] == "white" and self.ModeRotation == True:
                self.ModeRotation = True
                self.addAngle = 0
                self.activeQuad[1] = ""
            elif self.activeQuad[1] == "red" and self.activeQuad[2] == "white" and self.ModeRotation == False:
                self.ModeRotation = False
                self.addAngle = -180  # !!!!!!!!!!!

def CheckRotation(self):
    last = self.Angle
    self.Angle = int(round(GetAngle(self, self.temp_loc_last, self.temp_loc_first,
                                    self.V_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)))))
    self.realAngle = self.Angle
    if self.Step > 1:
        negativ = False
        if self.Angle < 0:
            negativ = True
            self.Angle = self.Angle * -1
        rain = round((self.Angle / self.Step)) * self.Step
        if rain in self.steps and rain != last:
            self.Angle = rain
            if negativ:
                self.Angle = self.Angle * -1
            return True
        else:
            self.Angle = last
            return False
    else:
        if last != self.Angle:
            return True
        else:
            return False

def SetupAxisUV(self):
    x = abs(self.temp_loc_first[0] - self.temp_loc_last[0])
    y = abs(self.temp_loc_first[1] - self.temp_loc_last[1])

    if x > y:
        return 'x'
    else:
        return 'y'

def CalculatePointForStartDrawing(self):
    context = bpy.context
    self.width = 0
    for i in bpy.context.screen.areas:
        if i.type == 'VIEW_3D':
            for j in i.regions:
                if j.type == 'UI':
                    width = j.width
    self.arr = []
    Offset = context.region.height * 0.05
    self.arr.append(V((0, 0)))
    self.arr.append(V((bpy.context.region.width - width, 0)))
    self.arr.append(V((context.region.width - width, context.region.height * 0.05)))
    self.arr.append(V((context.region.width - width, context.region.height * 0.05)))
    self.arr.append(V((0, context.region.height * 0.05)))
    self.arr.append(V((0, 0)))

    self.arr.append(V((context.region.width - width, context.region.height * 0.05)))
    self.arr.append(V((bpy.context.region.width - width, context.region.height)))
    self.arr.append(V(((context.region.width - width) * 0.95, context.region.height)))
    self.arr.append(V(((context.region.width - width) * 0.95, context.region.height)))
    self.arr.append(V(((context.region.width - width) * 0.95, context.region.height * 0.05)))
    self.arr.append(V(((context.region.width - width), context.region.height * 0.05)))

    self.arr.append(V(((context.region.width - width) * 0.95, context.region.height)))
    self.arr.append(V(((context.region.width - width) * 0.05, context.region.height)))
    self.arr.append(V(((context.region.width - width) * 0.05, context.region.height * 0.95)))
    self.arr.append(V(((context.region.width - width) * 0.05, context.region.height * 0.95)))
    self.arr.append(V(((context.region.width - width) * 0.95, context.region.height * 0.95)))
    self.arr.append(V(((context.region.width - width) * 0.95, context.region.height)))

    self.arr.append(V((0, context.region.height * 0.05)))
    self.arr.append(V(((context.region.width - width) * 0.05, context.region.height * 0.05)))
    self.arr.append(V(((context.region.width - width) * 0.05, context.region.height)))
    self.arr.append(V(((context.region.width - width) * 0.05, context.region.height)))
    self.arr.append(V((0, context.region.height)))
    self.arr.append(V((0, context.region.height * 0.05)))

    self.arr.append(V((0, context.region.height - Offset)))
    self.width = width

def DrawStartTool(self, context):
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'TRIS', {"pos": self.arr})
        shader.uniform_float("color", (1, 1, 1, 0.1))
        batch.draw(shader)

        blf.position(0, ((context.region.width - self.width) / 2) - context.region.width * 0.1,
                     (context.region.height * 0.025), 0)
        blf.size(0, 30, 50)
        blf.draw(0, str(self.toolName))
    except:
        pass

def DrawSCallBackZero(self, context):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": self.mergeLine})
    shader.uniform_float("color", self.color)
    batch.draw(shader)

    batch = batch_for_shader(shader, 'TRIS', {"pos": self.arrowLeft})
    shader.uniform_float("color", self.color)
    batch.draw(shader)

    batch = batch_for_shader(shader, 'TRIS', {"pos": self.arrowRight})
    shader.uniform_float("color", self.color)
    batch.draw(shader)

    batch = batch_for_shader(shader, 'LINES', {"pos": self.conturArrow})
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.4))
    batch.draw(shader)

    batch = batch_for_shader(shader, 'LINES', {"pos": self.conturLine})
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.4))
    batch.draw(shader)

    if self.line != None:
        batch = batch_for_shader(shader, 'LINES', {"pos": self.line})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
        batch.draw(shader)

        blf.position(0, self.mousePosition[0], self.mousePosition[1], 0)
        blf.size(0, 30, 50)
        blf.draw(0, str(self.axis))

def CalculateDrawZero(self, context):
    for i in bpy.context.window.screen.areas:
        if i.type == 'VIEW_3D':
            CD = i.spaces[0].region_3d.view_distance / 10
            if CD >= 0.3:
                MS = Matrix.Scale(CD, 4, )
            else:
                MS = Matrix.Scale(0.3, 4, )

    MR = Matrix.Rotation(1.570796 * 2, 3, Vector((0.0, 0.0, -1.0))).to_4x4()
    MTL = Matrix.Translation(V((0, -1.1, 0)))
    MTR = Matrix.Translation(V((0, 1.1, 0)))

    v1 = V((0, -1, 0))
    v2 = V((1, 0, 0))
    v3 = V((0.4, 0, 0))
    v4 = V((-0.4, 0, 0))
    v5 = V((-1, -0, 0))
    v6 = V((0.4, 2.2, 0))
    v7 = V((-0.4, 2.2, 0))

    v8 = V((4, -0.1, 0))
    v9 = V((4, 0.1, 0))
    v10 = V((-4, 0.1, 0))
    v11 = V((-4, -0.1, 0))

    arrow = [v1, v2, v3,
             v1, v5, v4,
             v4, v3, v1,
             v3, v6, v7,
             v4, v3, v7]

    merge = [v8, v9, v10, v10, v11, v8]

    self.conturArrow_X_L = []
    self.conturArrow_Y_L = []
    self.conturArrow_Z_L = []

    self.conturArrow_X_R = []
    self.conturArrow_Y_R = []
    self.conturArrow_Z_R = []

    self.conturLine_X = []
    self.conturLine_Y = []
    self.conturLine_Z = []

    self.conturArrow = []
    self.conturLine = []

    conturArrow = [v1, v2, v2, v3, v3, v6, v6, v7, v7, v4, v4, v5, v5, v1]
    conturLine = [v8, v9, v9, v10, v10, v11, v11, v8]

    self.arrowLeft = []
    self.arrowRight = []
    self.mergeLine = []

    self.arrowLeft_X = []
    self.arrowRight_X = []
    self.mergeLine_X = []

    self.arrowLeft_Y = []
    self.arrowRight_Y = []
    self.mergeLine_Y = []

    self.arrowLeft_Z = []
    self.arrowRight_Z = []
    self.mergeLine_Z = []

    if self.exc_axis == 'x':
        # if self.axis == 'x':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

        for i in merge:
            self.mergeLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.ZeroMatrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_X_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_X_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))

        # elif self.axis == 'y':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

        for i in merge:
            self.mergeLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.ZeroMatrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_Y_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_Y_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))
        # elif self.axis == 'z':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0, 0, 1)))).to_4x4()

        for i in merge:
            self.mergeLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.ZeroMatrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_Z_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_Z_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))

    if self.exc_axis == 'y':
        # if self.axis == 'x':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()

        for i in merge:
            self.mergeLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.ZeroMatrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_X_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_X_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))

        # elif self.axis == 'y':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

        for i in merge:
            self.mergeLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.ZeroMatrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_Y_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_Y_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))
        # elif self.axis == 'z':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0, 0, 0)))).to_4x4()

        for i in merge:
            self.mergeLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.ZeroMatrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_Z_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_Z_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))

    if self.exc_axis == 'z':
        # if self.axis == 'x':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()

        for i in merge:
            self.mergeLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.ZeroMatrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_X_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_X_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))

        # elif self.axis == 'y':
        self.ZeroMatrix = self.V_matrix  # @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()

        for i in merge:
            self.mergeLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.V_matrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.V_matrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.V_matrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_Y_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_Y_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))
        # elif self.axis == 'z':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0, 0, 1)))).to_4x4()

        for i in merge:
            self.mergeLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          ((self.ZeroMatrix @ MS)) @ i))
        for i in conturLine:
            self.conturLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS)) @ i))
        for i in arrow:
            self.arrowLeft_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                          (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.arrowRight_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                           ((self.ZeroMatrix @ MS) @ MTR) @ i))
        for i in conturArrow:
            self.conturArrow_Z_L.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      (((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
            self.conturArrow_Z_R.append(
                view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                      ((self.ZeroMatrix @ MS) @ MTR) @ i))

def CalculateDrawMirror(self, context):
    for i in bpy.context.window.screen.areas:
        if i.type == 'VIEW_3D':
            CD = i.spaces[0].region_3d.view_distance / 20
            if CD >= 0.3:
                MS = Matrix.Scale(CD, 4, )
            else:
                MS = Matrix.Scale(0.3, 4, )

    v1 = V((0.0, -1.0, 0.0))
    v2 = V((1.0, 0.0, 0.0))
    v3 = V((0.4, 0.0, 0.0))
    v4 = V((-0.4, 0.0, 0.0))
    v5 = V((-1.0, -0.0, 0.0))
    v6 = V((0.4, 3.0, 0.0))
    v7 = V((-0.4, 3.0, 0.0))

    arrow = [v1, v2, v3,
             v1, v5, v4,
             v4, v3, v1,
             v3, v6, v7,
             v4, v3, v7]
    conturArrow = [v1, v2, v2, v3, v3, v6, v6, v7, v7, v4, v4, v5, v5, v1]

    MR = Matrix.Rotation(1.570796 * 2, 3, Vector((0.0, 0.0, 1.0))).to_4x4()
    MT = Matrix.Translation(V((1.05, -1.7, 1.05)))
    a = []
    for i in arrow:
        a.append(MT @ i)
    arrow = a
    a = []
    for i in conturArrow:
        a.append(MT @ i)
    conturArrow = a
    a = []
    MT = Matrix.Translation(V((-1.05 * 2, -1.5 * 2, 0)))
    for i in arrow:
        a.append((MR @ MT) @ i)
    arrow += a
    a = []
    for i in conturArrow:
        a.append((MR @ MT) @ i)
    conturArrow += a
    a = []

    v8 = V((1, 3.0 / 2, 1))
    v9 = V((-1, 3.0 / 2, 1))
    v10 = V((-1, 3.0 / 2, -1))
    v11 = V((1, 3.0 / 2, -1))

    merge = [v8, v9, v10, v10, v11, v8]
    conturLine = [v8, v9, v9, v10, v10, v11, v11, v8]

    MT = Matrix.Translation(V((2.1, 0, 0)))
    a = []
    for i in merge:
        a.append(MT @ i)
    merge += a
    a = []
    for i in conturLine:
        a.append(MT @ i)
    conturLine += a
    a = []
    MT = Matrix.Translation(V((0, 0, 2.1)))
    a = []
    for i in merge:
        a.append(MT @ i)
    merge += a
    a = []
    for i in conturLine:
        a.append(MT @ i)
    conturLine += a

    self.countr = []
    self.arrow = []

    self.countr_X = []
    self.countr_Y = []
    self.countr_Z = []

    self.arrow_X = []
    self.arrow_Y = []
    self.arrow_Z = []

    AM = arrow + merge
    MT = Matrix.Translation(V((-1.05, -1.5, -1.05)))
    C = conturLine + conturArrow

    if self.exc_axis == 'x':
        # if self.axis == 'x':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()
        for i in AM:
            self.arrow_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))

        # elif self.axis == 'y':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

        for i in AM:
            self.arrow_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))
        # elif self.axis == 'z':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()

        for i in AM:
            self.arrow_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))

    if self.exc_axis == 'y':
        # if self.axis == 'x':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()
        for i in AM:
            self.arrow_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))

        # elif self.axis == 'y':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

        for i in AM:
            self.arrow_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))
        # elif self.axis == 'z':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

        for i in AM:
            self.arrow_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))

    if self.exc_axis == 'z':
        # if self.axis == 'x':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()
        for i in AM:
            self.arrow_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))

        # elif self.axis == 'y':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

        for i in AM:
            self.arrow_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))
        # elif self.axis == 'z':
        self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()

        for i in AM:
            self.arrow_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                      (((self.ZeroMatrix @ MS)) @ MT) @ i))
        for i in C:
            self.countr_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                                       (((self.ZeroMatrix @ MS)) @ MT) @ i))

def DrawScaleMirror(self, context):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": self.arrow})
    shader.uniform_float("color", self.color)
    batch.draw(shader)

    batch = batch_for_shader(shader, 'LINES', {"pos": self.countr})
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.4))
    batch.draw(shader)

    if self.line != None:
        batch = batch_for_shader(shader, 'LINES', {"pos": self.line})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
        batch.draw(shader)

        blf.position(0, self.mousePosition[0], self.mousePosition[1], 0)
        blf.size(0, 30, 50)
        blf.draw(0, str(self.axis))

class ActionsState():
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
    RUNNING_MODAL = {'INTERFACE'}

class AdvancedTransform(Operator):
    ''' Advanced move '''
    bl_idname = "view3d.advancedtransform"
    bl_label = "Advanced Transform"

    def __init__(self):
        print("Start")
        self.Toolname = ""
        self.header_text = ""
        self.SkipFrameValue = 2
        self.SkipFrameCount = 2
        self.OldMousePos = mathutils.Vector((0.0,0.0,0.0))
        self.NewMousePos = mathutils.Vector((0.0,0.0,0.0))
        self.Expected_Action = None # Finction be calling after delay
        self.PivotPoint = None
        self.TransfromOrientationMatrix = None
        self.NormalIntersectionPlane = None

        self.Event = bpy.types.Event # Temp Variable for saving event



        self.ActionsState = ActionsState()
        self.GenerationGenerateLambdaEvents()
        self.GenerateDelegates()

        self.GML = lambda event: GetMouseLocation(self.PivotPoint,self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event)

    def GenerationGenerateLambdaEvents(self):
        self.If_Modify = lambda event: event.shift or event.alt

        self.If_LM = lambda event: event.type == 'LEFTMOUSE'
        self.If_LM_Cond = lambda event: (self.If_LM(event) or self.ActionsState.LeftMouse) and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_MMove = lambda event: event.type == 'MOUSEMOVE' 
        self.If_Pass = lambda event: (self.If_MMove(event) and self.ActionsState.Pass) and self.SkipFrameCount > 0

        self.If_MM = lambda event: event.type == 'MIDDLEMOUSE'
        self.If_MM_Cond = lambda event: self.If_MM(event) or self.ActionsState.MiddleMouse and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_RM = lambda event: event.type == 'RIGHTMOUSE'
        self.If_RM_Cond = lambda event: self.If_RM(event) or self.ActionsState.RightMouse and (self.If_Alt(event) != True and self.If_Shift(event) != True and self.If_Ctrl(event) != True)

        self.If_Spcae = lambda event: event.type == 'SPACE'
        self.If_Spcae_Cond = lambda event: self.If_Spcae(event) or self.ActionsState.Space

        self.If_Shift = lambda event: event.shift
        self.If_Shift_Cond = lambda event: self.If_Shift(event) or self.ActionsState.Shift

        self.If_Alt = lambda event: event.alt
        self.If_Alt_Cond = lambda event: self.If_Alt(event) or self.ActionsState.Alt

        self.If_Ctrl = lambda event: event.ctrl
        self.If_Ctrl_Cond = lambda event: self.If_Ctrl(event) or self.ActionsState.Ctrl

        self.If_Esc = lambda event: event.type == 'ESC'
        self.If_Esc_Cond = lambda event: self.If_Esc(event) or self.ActionsState.Esc

        self.If_G = lambda event: event.unicode == 'G' or event.unicode == 'g'
        self.If_X = lambda event: event.unicode == 'X' or event.unicode == 'x'
        self.If_Y = lambda event: event.unicode == 'Y' or event.unicode == 'y'
        self.If_Z = lambda event: event.unicode == 'Z' or event.unicode == 'z'

    def GenerateDelegates(self):
        # -----------------------LEFT_MOUSE Only Axis Move---------------------------------------------------#
        self.LM_D = lambda event: self.StartDelay(event, self.ActionsState.LeftMouse, self.AfterLeftMouseAction, self.BedoreLeftMouseAction)
        # -----------------------RIGHT_MOUSE Exlude Axis-----------------------------------------------------#
        self.RM_D = lambda event: self.StartDelay(event, self.ActionsState.RightMouse, self.AfterRightMouseAction, self.BeforeRightMouseAction, 0)
        # -----------------------MIDDLE_MOUSE No Constrain---------------------------------------------------#
        self.MM_D = lambda event: self.StartDelay(event, self.ActionsState.MiddleMouse, self.AfterMiddleMouseAction, self.BeforeMiddleMouseAction, 0)
        # -----------------------MOVE_MOUSE No Constrain---------------------------------------------------#
        self.MoveM_D = lambda event: self.StartDelay(event, self.ActionsState.MiddleMouse, self.AfterMiddleMouseAction, self.BeforeMiddleMouseAction, 0)
        # -----------------------SPACEBAR Bottom-------------------------------------------------------------#
        self.Space_D = lambda event: self.StartDelay(event, self.ActionsState.Space, self.AfterSpaceAction, self.BeforeSpaceAction, 0)
        #----------------------Tweak with new selection------------------------------------------------------#
        self.Shift_D = lambda event: self.StartDelay(event, self.ActionsState.Shift, self.AfterShiftAction , self.BeforeShiftAction)    
        #----------------------Tweak with old selection------------------------------------------------------#
        self.Alt_D = lambda event: self.StartDelay(event, self.ActionsState.Alt, self.AfterAltAction , self.BeforeAltAction)

    # We can use actions before delay and after delay
    def BedoreLeftMouseAction(self, event):
        print("BedoreLeftMouseAction")
    def AfterLeftMouseAction(self, event):
        print("AfterLeftMouseAction")
        return {'FINISHED'}
    def BeforeRightMouseAction(self, event):
        print("BeforeRightMouseAction")
    def AfterRightMouseAction(self, event):
        print("AfterRightMouseAction")
        return {'FINISHED'}
    def BeforeMiddleMouseAction(self, event):
        print("BeforeMiddleMouseAction")
    def AfterMiddleMouseAction(self, event):
        print("AfterMiddleMouseAction")
        return {'FINISHED'}
    def BeforeMoveMouseAction(self, event):
        print("BeforeMoveMouseAction")
        return {'FINISHED'}
    def AfterMoveMouseAction(self, event):
        print("AfterMoveMouseAction")
        return {'FINISHED'}
    def BeforeSpaceAction(self, event):
        print("BeforeSpaceAction")
    def AfterSpaceAction(self, event):
        print("AfterSpaceAction")
        return {'FINISHED'}
    def BeforeShiftAction(self, event):
        print("BeforeShiftAction")
    def AfterShiftAction(self, event):
        self.Expected_Action = None
        print("AfterShiftAction")
        return {'RUNNING_MODAL'}
    def BeforeAltAction(self, event):
        print("BeforeAltAction")
    def AfterAltAction(self, event):
        print("BeforeAltAction")
        return {'RUNNING_MODAL'}
    def BeforeCtrlAction(self, event):
        print("BeforeCtrlAction")
    def AfterCtrlAction(self, event):
        print("AfterCtrlAction")
        return {'RUNNING_MODAL'}
    def Before_G(self, event):
        print("Before_G")
    def After_G(self, event):
        return {'FINISHED'}
    def Before_X(self, event):
        print("Before_X")
    def After_X(self, event):
        print("After_X")
        return {'FINISHED'}
    def Before_Y(self, event):
        print("Before_Y")
    def After_Y(self, event):
        print("After_Y")
        return {'FINISHED'}
    def Before_Z(self, event):
        print("Before_Z")
    def After_Z(self, event):
        print("After_Z")
        return {'FINISHED'}

    def Pass(self, event):
        pass

    def SaveEvent(self, event):
        self.Event.alt = event.alt
        self.Event.ascii = event.ascii
        self.Event.ctrl = event.ctrl
        self.Event.direction = event.direction
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

    def SetUp(self, event):
        UserSnap(self)
        UserPresets(self)
        self.PivotPoint = GetPivotPointPoistion()
        self.TransfromOrientationMatrix = GetTransfromOrientationMatrix()
        self.NormalIntersectionPlane = GetNormalForIntersectionPlane(self.TransfromOrientationMatrix)
        #self.OldMousePos = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
        #CalculatePointForStartDrawing(self)

    def ClearViewport(self):
        # if context.preferences.addons[__name__].preferences.Draw_Tooll:
        #     bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
        pass

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "VIEW_3D"
    
    def StartDelay(self, event, State: ActionsState, ActionAfterDelay, ActionBeforeDelay, SkipFrames = 4):
        self.SkipFrameCount = SkipFrames
        self.TransfromOrientationMatrix = GetTransfromOrientationMatrix()
        self.PivotPoint = GetPivotPointPoistion()
        self.NormalIntersectionPlane = GetNormalForIntersectionPlane(self.TransfromOrientationMatrix)
        self.OldMousePos = self.GML(event)
        self.ActionsState.Pass = True
        State = True
        ActionBeforeDelay(event)
        self.SaveEvent(event)
        if ActionBeforeDelay:
            self.Expected_Action = ActionAfterDelay
        if SkipFrames == 0:
            self.Expected_Action(event)
            return {'FINISHED'}
        else:
            return {'RUNNING_MODAL'}
          
    def modal(self, context, event):
        # -----------------------Skip Frames-------------------------------------------------------------#
        if self.If_Pass(event):
            self.SkipFrameCount -= 1
            return {'RUNNING_MODAL'}
        elif self.Expected_Action: 
            self.ActionsState.Pass = False
            self.NewMousePos = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
            return self.Expected_Action(self.Event)

        # -----------------------LEFT_MOUSE Only Axis Move---------------------------------------------------#
        if self.If_LM_Cond(event):
            return self.LM_D(event)
        # -----------------------RIGHT_MOUSE Exlude Axis-----------------------------------------------------#
        elif self.If_RM_Cond(event):
            return self.RM_D(event)
        # -----------------------MIDDLE_MOUSE No Constrain---------------------------------------------------#
        elif self.If_MM_Cond(event):
            return self.MM_D(event)
        # -----------------------SPACEBAR Bottom-------------------------------------------------------------#
        elif self.If_Spcae_Cond(event):
            return self.Space_D(event)
        #----------------------Tweak with new selection------------------------------------------------------#
        elif self.If_Shift_Cond(event):
            return self.Shift_D(event)
        #----------------------Tweak with old selection------------------------------------------------------#
        elif self.If_Alt_Cond(event):
            return self.Alt_D(event)

        if self.If_G(event):
            return self.StartDelay(event, self.ActionsState.G, self.After_G, self.Before_G, 0)
        if self.If_X(event):
            return self.StartDelay(event, self.ActionsState.X, self.After_X, self.Before_X, 0)
        if self.If_Y(event):
            return self.StartDelay(event, self.ActionsState.Y, self.After_Y, self.Before_Y, 0)
        if self.If_Z(event):
            return self.StartDelay(event, self.ActionsState.Z, self.After_Z, self.Before_Z, 0)

        if event.type == 'ESC':
            # if context.preferences.addons[__name__].preferences.Draw_Tooll:
            #     bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
            # context.area.header_text_set()
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        context.area.header_text_set(self.header_text)
        if context.space_data.type == 'VIEW_3D':
            self.SetUp(event)
            # if context.preferences.addons[__name__].preferences.Draw_Tooll:
            #     argsStart = (self, context)
            #     self._handle1 = bpy.types.SpaceView3D.draw_handler_add(DrawStartTool, argsStart, 'WINDOW', 'POST_PIXEL')

            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

class AdvancedMove(AdvancedTransform):
    ''' Advanced move '''
    bl_idname = "view3d.advancedmove"
    bl_label = "Advanced Move"

    def __init__(self):
        super().__init__()
        self.header_text = 'Drag LMB constraint axis, RMB translate by two axis, MMB free translate, SPACE free translate with snap and rotate along normal, Shift tweak new selection, Alt Current selection'
        self.toolName = "Advanced Move"
        self.If_Alt_Cond =   lambda event:   self.If_Alt(event) and (self.If_LM(event) or self.If_RM(event) or self.If_MM(event))
        self.If_Shift_Cond = lambda event: self.If_Shift(event) and (self.If_LM(event) or self.If_RM(event) or self.If_MM(event))

    def AfterLeftMouseAction(self, event):
        obj = bpy.data.objects["Plane"]
        obj.matrix_world = ((self.NewMousePos - self.OldMousePos).normalized()).to_track_quat('Z', 'Y').to_matrix().to_4x4()
        obj.location = self.NewMousePos

        axis = GetMouseDirection(self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        SetConstarin.SetMoveOnlyOneAxis(axis)
        UserPresets(self, True)
        return {'FINISHED'}
    def AfterRightMouseAction(self, event):
        SetConstarin.SetMoveExclude(GetAxixForTransformation(self.NormalIntersectionPlane))
        UserPresets(self, True)
        return {'FINISHED'}
    def AfterMiddleMouseAction(self, event):
        SetConstarin.SetMoveNoConstrainNoSnap()
        if not event.is_repeat:
            UserPresets(self, False)
            return {'FINISHED'}
    def AfterSpaceAction(self, event):
        bpy.context.scene.tool_settings.snap_elements = {'FACE'}
        bpy.context.scene.tool_settings.snap_target = 'CENTER'
        bpy.context.scene.tool_settings.use_snap_align_rotation = True
        bpy.context.scene.tool_settings.snap_elements_individual = {'FACE_PROJECT'}
        bpy.context.scene.tool_settings.use_snap = True
        SetConstarin.SetMoveNoConstrain()
        return {'RUNNING_MODAL'}
    def AfterShiftAction(self, event):
        if context.mode == "EDIT_MESH":
            bpy.ops.mesh.select_all(action='DESELECT')
        else:
            bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.view3d.select('INVOKE_DEFAULT', extend=True, deselect=False, enumerate=False, toggle=False)
        if self.If_LM(event):
            print("##################Twik Left")
            self.AfterLeftMouseAction(event)
        if self.If_RM(event):
            print("##################Twik Right")
            self.AfterRightMouseAction(event)
        if self.If_MM(event):
            print("##################Twik Midl")
            self.AfterMiddleMouseAction(event)
        
        print("##################Twik Modae")
        self.ActionsState.Shift = False
        self.Expected_Action = None
        #bpy.ops.view3d.advancedmove()
        return {'RUNNING_MODAL'}
    def AfterAltAction(self, event):
        if self.If_LM(event):
            self.AfterLeftMouseAction(event)
        if self.If_RM(event):
            self.AfterRightMouseAction(event)
        if self.If_MM(event):
            self.AfterMiddleMouseAction(event)
        self.ActionsState.Alt = False
        self.Expected_Action = None
        #bpy.ops.view3d.advancedmove()
        return {'RUNNING_MODAL'}
    def After_G(self, event):
        if context.mode == "EDIT_MESH":
            if context.tool_settings.mesh_select_mode[1]:
                bpy.ops.transform.edge_slide('INVOKE_DEFAULT')
            else:
                bpy.ops.transform.vert_slide("INVOKE_DEFAULT")

        UserPresets(self, True)
        return {'FINISHED'}
    def After_X(self, event):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
        UserPresets(self, True)
        if context.preferences.addons[__name__].preferences.Draw_Tooll:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
        # context.area.header_text_set()
        return {'FINISHED'}
    def After_Y(self, event):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
        UserPresets(self, True)
        if context.preferences.addons[__name__].preferences.Draw_Tooll:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
        # context.area.header_text_set()
        return {'FINISHED'}
    def After_Z(self, event):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
        UserPresets(self, True)
        if context.preferences.addons[__name__].preferences.Draw_Tooll:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
        # context.area.header_text_set()
        return {'FINISHED'}

class AdvancedScale(AdvancedTransform):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale"
    bl_label = "Advanced Scale"

    def __init__(self):
        super().__init__()
        self.header_text = 'Drag LMB constraint axis, RMB resize by two axis, MMB free resize, SHIFT mirror, SPACE flatten'
        self.toolName = "Advanced Scale"
        self.If_Alt_Cond =   lambda event:   self.If_Alt(event) and (self.If_LM(event) or self.If_RM(event) or self.If_MM(event))
        self.If_Shift_Cond = lambda event: self.If_Shift(event) and (self.If_LM(event) or self.If_RM(event) or self.If_MM(event))

    def AfterLeftMouseAction(self, event):
        axis = GetMouseDirection(self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        SetConstarin.SetScaleOnly(axis)
        UserPresets(self, True)
        self.ClearViewport()
        return {'FINISHED'}
    
    def BeforeRightMouseAction(self, event):
        SetConstarin.SetScaleExclude(GetAxixForTransformation(self.NormalIntersectionPlane))
        UserPresets(self, True)
        self.ClearViewport()
        return {'FINISHED'}
    
    def AfterSpaceAction(self, event):
        bpy.ops.view3d.advancedscale_zero('INVOKE_DEFAULT')
        UserPresets(self, True)
        self.ClearViewport()
        return {'FINISHED'}
    
    def AfterMiddleMouseAction(self, event):
        SetConstarin.SetScaleNoConstrain()
        UserPresets(self, True)
        self.ClearViewport()
        return {'FINISHED'}
    
    def AfterShiftAction(self, event):
        bpy.ops.view3d.advancedscale_mirror('INVOKE_DEFAULT')
        UserPresets(self, True)
        self.ClearViewport()
        return {'FINISHED'}
    
    def After_X(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, False))
        UserPresets(self, True)
        self.ClearViewport()
        return {'FINISHED'}
    def After_Y(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, False))
        UserPresets(self, True)
        self.ClearViewport()
        return {'FINISHED'}
    def After_Z(self, event):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, False, True))
        UserPresets(self, True)
        self.ClearViewport()
        return {'FINISHED'}

class AdvancedScaleZore(Operator):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale_zero"
    bl_label = "Advanced Scale zero"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "VIEW_3D"

    def modal(self, context, event):
        self.temp_loc_last = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
        self.mousePosition = event.mouse_region_x, event.mouse_region_y
        self.line = [(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)),
                     (view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                            self.temp_loc_last))]
        self.axis = ChouseAxisByMouseDirection(self, self.center, self.temp_loc_last)
        if self.axis == 'x':
            self.arrowLeft = self.arrowLeft_X
            self.arrowRight = self.arrowRight_X
            self.mergeLine = self.mergeLine_X
            self.conturArrow = self.conturArrow_X_R + self.conturArrow_X_L
            self.conturLine = self.conturLine_X

        elif self.axis == 'y':
            self.arrowLeft = self.arrowLeft_Y
            self.arrowRight = self.arrowRight_Y
            self.mergeLine = self.mergeLine_Y
            self.conturArrow = self.conturArrow_Y_R + self.conturArrow_Y_L
            self.conturLine = self.conturLine_Y

        elif self.axis == 'z':
            self.arrowLeft = self.arrowLeft_Z
            self.arrowRight = self.arrowRight_Z
            self.mergeLine = self.mergeLine_Z
            self.conturArrow = self.conturArrow_Z_R + self.conturArrow_Z_L
            self.conturLine = self.conturLine_Z

        context.area.tag_redraw()
        if event.type == 'LEFTMOUSE':
            SetConstarin.SetScaleOnlySetZero(self, context, self.axis)
            UserPresets(self, True)
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_Zero, 'WINDOW')
            context.area.tag_redraw()
            return {'FINISHED'}

        if event.type == 'ESC':
            UserPresets(self, True)
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_Zero, 'WINDOW')
            context.area.tag_redraw()
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        # context.area.header_text_set('Drag LMB constraint axis, RMB resize by two axis, MMB free resize, SHIFT mirror, SPACE flatten')
        if context.space_data.type == 'VIEW_3D':
            UserSnap(self, context)
            UserPresets(self, context)
            GetTransfromOrientationMatrix(self)
            GetPivotPointPoistion(self)
            GetNormalForIntersectionPlane(self)

            self.temp_loc_last = GetMouseLocation(self, event)
            self.mousePosition = event.mouse_region_x, event.mouse_region_y
            self.line = [
                (view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)),
                (view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.temp_loc_last))
            ]
            self.axis = ChouseAxisByMouseDirection(self, self.center, self.temp_loc_last)
            CalculateDrawZero(self, context)
            self.line = None
            self.color = V((0.8, 0.4, 0.0, 0.3))

            argsStart = (self, context)
            self._handle_Zero = bpy.types.SpaceView3D.draw_handler_add(DrawSCallBackZero, argsStart, 'WINDOW',
                                                                       'POST_PIXEL')

            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

class AdvancedScaleMirror(Operator):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale_mirror"
    bl_label = "Advanced Scale Mirror"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "VIEW_3D"

    def modal(self, context, event):
        self.temp_loc_last = GetMouseLocation(self, event)
        self.mousePosition = event.mouse_region_x, event.mouse_region_y
        self.line = [(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)),
                     (view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
                                                            self.temp_loc_last))]
        self.axis = ChouseAxisByMouseDirection(self, self.center, self.temp_loc_last)
        if self.axis == 'x':
            self.arrow = self.arrow_X
            self.countr = self.countr_X

        elif self.axis == 'y':
            self.arrow = self.arrow_Y
            self.countr = self.countr_Y

        elif self.axis == 'z':
            self.arrow = self.arrow_Z
            self.countr = self.countr_Z

        context.area.tag_redraw()
        if event.type == 'LEFTMOUSE':
            SetConstarin.SetScaleOnlySetNegative(self, context, self.axis)
            UserPresets(self, True)
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_Mirror, 'WINDOW')
            context.area.tag_redraw()
            return {'FINISHED'}

        if event.type == 'ESC':
            UserPresets(self, True)
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_Mirror, 'WINDOW')
            context.area.tag_redraw()
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        # context.area.header_text_set('Drag LMB constraint axis, RMB resize by two axis, MMB free resize, SHIFT mirror, SPACE flatten')
        if context.space_data.type == 'VIEW_3D':
            UserSnap(self, context)
            UserPresets(self, context)
            GetTransfromOrientationMatrix(self)
            GetPivotPointPoistion(self)
            GetNormalForIntersectionPlane(self)

            self.temp_loc_last = GetMouseLocation(self, event)
            self.mousePosition = event.mouse_region_x, event.mouse_region_y
            self.line = [
                (view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)),
                (view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.temp_loc_last))
            ]
            self.axis = ChouseAxisByMouseDirection(self, self.center, self.temp_loc_last)
            CalculateDrawMirror(self, context)
            self.line = None
            self.color = V((0.8, 0.4, 0.0, 0.3))

            argsStart = (self, context)
            self._handle_Mirror = bpy.types.SpaceView3D.draw_handler_add(DrawScaleMirror, argsStart, 'WINDOW',
                                                                         'POST_PIXEL')

            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

class AdvancedRotation(AdvancedTransform):
    """ Advanced move """
    bl_idname = "view3d.advanced_rotation"
    bl_label = "Advanced Rotation"
    #bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        super().__init__()
        self.header_text = 'LMB constraint view axis, RMB constraint view axis snap, MMB free rotate'
        self.ToolName = 'Advanced Rotation'
        self.MouseDirection = mathutils.Vector((0.0,0.0,0.0))
        self.RotationValue = 0.0

        self.IsFirstIteration = True
        self.IsIncreaseAngle = False
        self.SwitchDirection = lambda DotProd: (not self.IsIncreaseAngle) if DotProd < 0 else self.IsIncreaseAngle
        self.AngleSnappingStep = int(context.preferences.addons[__name__].preferences.Snapping_Step)

        



    @classmethod
    def poll(cls, context):
        return context.space_data.type == "VIEW_3D"

    def GetAngle(self, v1, v2):
        v1 = v1 - self.PivotPoint
        v2 = v2 - self.PivotPoint
        # we can get "ValueError: math domain error" or ZeroDivisionError:
        try:
            cos_angle = v1.dot(v2) / (v1.length * v2.length)
            angle = math.acos(cos_angle)
            return math.degrees(angle)
        except:
            deb("ERRRRRROOOOORR")
            return 0

    def AfterLeftMouseAction(self, event):
        self.ActionsState.MoveMouse = True
        return 
    def AfterMoveMouseAction(self, event):
        self.NewMousePos = self.GML(event)
        angle = self.GetAngle(self.OldMousePos, self.NewMousePos)

        if round(angle / self.AngleSnappingStep) * self.AngleSnappingStep != 0:
            # find third axis 1 is mouse direction 2 is view direction  and 3 (corss) look at pivot point
            cross=((self.NewMousePos - self.OldMousePos).normalized()).cross(self.NormalIntersectionPlane)
            # if value biger then 0 counterclock-wise else clockwise
            angle = angle*-1 if cross.dot(self.NewMousePos - self.PivotPoint) > 0 else angle

            self.OldMousePos = self.NewMousePos.copy()
            self.RotationValue += angle
            
            SetConstarin.SetRotationOnlyAT(angle , GetAxixForTransformation(self.NormalIntersectionPlane))
        return 

    def modal(self, context, event):
        if self.If_LM(event):
            self.LM_D(event)
        
        if self.If_MMove(event) and self.ActionsState.LeftMouse:
            

        if event.type == 'ESC':
            #UserPresets(self, True)
            
            context.area.tag_redraw()
            return {'FINISHED'}
        return {'RUNNING_MODAL'}

    # def invoke(self, context, event):
    #     self.rot = bpy.context.active_object.rotation_euler

    #     #UserSnap(self)
    #     #UserPresets(self, context)
    #     GetTransfromOrientationMatrix(self)
    #     GetPivotPointPoistion(self)
    #     GetNormalForIntersectionPlane(self)
    #     self.OldMousePos = GetMouseLocation(self, event)

    #     context.window_manager.modal_handler_add(self)
    #     return {'RUNNING_MODAL'}
        # else:
        #     self.report({'WARNING'}, "Active space must be a View3d")
        return {'CANCELLED'}

class AdvancedMoveUV(Operator):
    ''' Advanced move '''
    bl_idname = "view3d.advancedmove_uv"
    bl_label = "Advanced Move UV"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "IMAGE_EDITOR"

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
                    SetConstarin.SetScaleOnlySetNegative(self, context, self.axis)
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
                    # print('SOSOK')
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
    def Axises(axis):
        return (axis == 'x', axis == 'y', axis == 'z')

    @staticmethod
    def SetMoveOnlyOneAxis(axis):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis = SetConstarin.Axises(axis))
    @staticmethod
    def SetMoveExclude(axis):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SetMoveNoConstrain():
        bpy.ops.transform.translate('INVOKE_DEFAULT')
    @staticmethod
    def SetMoveNoConstrainNoSnap():
        bpy.ops.transform.translate('INVOKE_DEFAULT')
    @staticmethod
    def SetRotationOnlyAT(value, axis):
        bpy.ops.transform.rotate(value=math.radians(value), constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SetRotationOnly(axis):
        bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SetScaleOnly(axis):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SetScaleExclude(axis):
        bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SetScaleNoConstrain():
        bpy.ops.transform.resize('INVOKE_DEFAULT')
    @staticmethod
    def SetScaleOnlySetZero(axis):
        axis_value = {'x': (0.0, 1.0, 1.0), 'y': (1.0, 0.0, 1.0), 'z': (1.0, 1.0, 0.0)}
        bpy.ops.transform.resize(value=axis_value[axis], constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SetScaleExcludeSetZero(axis):
        axis_value = {'x': (0.0, 1.0, 1.0), 'y': (1.0, 0.0, 1.0), 'z': (1.0, 1.0, 0.0)}
        axis_mapping = {'x': (False, True, True), 'y': (True, False, True), 'z': (True, True, False)}
        bpy.ops.transform.resize(value=axis_value[axis], constraint_axis=axis_mapping[axis])
    @staticmethod
    def SetScaleOnlySetNegative(axis):
        axis_value = {'x': (-1.0, 1.0, 1.0), 'y': (1.0, -1.0, 1.0), 'z': (1.0, 1.0, -1.0)}
        bpy.ops.transform.resize(value=(-1.0, 1.0, 1.0), constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SnapRotation(axis, value):
        bpy.ops.transform.rotate(value=math.radians(value),constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SetMoveOnlyUV(axis):
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=SetConstarin.Axises(axis))
    @staticmethod
    def SetMoveExcludeUV():
        bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, True, False))


classes = (AdvancedTransform, AdvancedMove, AdvancedScale, AdvancedRotation, AdvancedMoveUV, AdvancedScaleUV, AdvancedRotationUV,
           AdvancedTransformPref, AdvancedScaleZore, AdvancedScaleMirror, AdvancedTransform_Add_Hotkey)


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
