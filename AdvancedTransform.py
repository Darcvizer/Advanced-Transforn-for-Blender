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
from bpy_extras import view3d_utils
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
        null.name = "ADHelpNull"
        bpy.ops.object.constraint_add(type='COPY_TRANSFORMS')
        bpy.context.object.constraints["Copy Transforms"].target = plane    
    if DubegMode:
        print(dis if dis != None and isinstance(dis, str) else "",': ' , value)
        if dis != None and isinstance(dis, Vector):
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
            else:
                CreateObjects()
        if forced:
            CreateObjects()
    return 0
                
def GetNormalForIntersectionPlane(TransfromOrientationMatrix):
    """Getting the closest axis to camera direction"""
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
    deb(matrix_transform_orientation)
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

def GetAxixForTransformation(normal):
    """Get axis by mouse direaction
    normal: vector"""
    index_of_max_value = max(range(len(normal)), key=lambda i: abs(normal[i])) # get index of maximal value at vector
    deb(normal)
    axies = ['x','y','z']
    return axies[index_of_max_value] # create array and get value by index

def GetMouseDirection(point_from, point_to, matrix):
    """Get mouse direction by 2 saved points, also if we have specific transform orientation we have to rotate vector by orientation matrix
    point_from: Vector | mouse point 1
    point_to: Vector | mouse point 2
    matrix: Matrix | Matrix for rotation vector
    """
    direction = (matrix.to_3x3().inverted() @ (point_from - point_to).normalized()) * -1 # Rotate move direction to transform orientation matrix
    #direction = (matrix.to_3x3() @ (point_from - point_to).normalized()) * -1
    return GetAxixForTransformation(direction)


def SetupAxisUV(self):
    x = abs(self.temp_loc_first[0] - self.temp_loc_last[0])
    y = abs(self.temp_loc_first[1] - self.temp_loc_last[1])

    if x > y:
        return 'x'
    else:
        return 'y'

def DrawScaleMirror(self, context):
    #try:
    if self.TransfromOrientationMatrix != None and self.MirrorAxis != None and self.PivotPoint != None:
        direction_axis_index = 0 if self.MirrorAxis == 'x' else (1 if self.MirrorAxis == 'y' else 2)
        direction_vector = self.TransfromOrientationMatrix[direction_axis_index].to_3d() 

        current_direction = V((+0.0, +1.0, 0.0)) if direction_axis_index == 1 else V((+0.0, +1.0, 0.0))*-1 # i don't know why but axies X and Z inverted and i must flip them
        angle = current_direction.angle(direction_vector) # get angle between original direction and desired rirection
        axis_for_rotation = current_direction.cross(direction_vector) # Axis for rotation

        # Rotate Vector By Transform Orientation Matrix
        RTM = lambda vector: (vector @ mathutils.Matrix.Rotation(angle, 4, axis_for_rotation)) + self.PivotPoint
        offset = V((0,3,0))
        # arrow vertices position with offset and alinge by matrix
        a_v1 = RTM(V((+0.0, +1.0, +0.0))) + offset
        a_v2 = RTM(V((-1.0, +0.0, +0.0))) + offset
        a_v3 = RTM(V((-0.4, +0.0, +0.0))) + offset
        a_v4 = RTM(V((-0.4, -2.0, +0.0))) + offset
        a_v5 = RTM(V((+0.4, -2.0, +0.0))) + offset
        a_v6 = RTM(V((+0.4, +0.0, +0.0))) + offset
        a_v7 = RTM(V((+1.0, +0.0, +0.0))) + offset

        p_v1 = RTM(V((+5.0, +0.0, +5.0)))
        p_v2 = RTM(V((+5.0, +0.0, -5.0)))
        p_v3 = RTM(V((-5.0, +0.0, -5.0)))
        p_v4 = RTM(V((-5.0, +0.0, +5.0)))

        # make triangles, each triange shuld veve 3 vertices. that method not use idexes only vertices order
        arrow_vertices =[a_v1, a_v2, a_v3,
                         a_v3, a_v4, a_v5,
                         a_v3, a_v5, a_v6,
                         a_v1, a_v6, a_v7,
                         a_v1, a_v3, a_v6]
        
        plane =         [p_v1, p_v2, p_v4, 
                         p_v2 ,p_v4, p_v3]
        # make loop for conture
        contur_arrow = [a_v1, a_v2, a_v3, a_v4, a_v5, a_v6, a_v7]
        contur_plane = [p_v1,p_v2,p_v3,p_v4]

        # Facing 
        ApplyFacing = lambda angle, dir, arr: [(i) @ mathutils.Matrix.Rotation(angle, 4, dir) for i in arr]
        SignedAngle= lambda v1, v2, axis: v1.angle(v2) * (1 if axis.dot(v1.cross(v2)) >= 0 else -1)

        direction_vector_camera = (bpy.context.region_data.view_rotation @ Vector((0, 0, -1))) # get view view direction
        up_direction = axis_for_rotation.cross(direction_vector)
        angle_to_camera = SignedAngle(up_direction, direction_vector_camera, direction_vector) * -1

        arrow_vertices = ApplyFacing(angle_to_camera, direction_vector, arrow_vertices)
        contur_arrow = ApplyFacing(angle_to_camera, direction_vector, contur_arrow)

        # Make shader and batsh
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.line_width_set(3)
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("ALPHA")

        batch = batch_for_shader(shader, 'TRIS', {"pos": plane})
        shader.uniform_float("color", (0.0, 0.3, 0.6, 0.3))
        batch.draw(shader)

        batch = batch_for_shader(shader, 'TRIS', {"pos": arrow_vertices})
        shader.uniform_float("color", (0.0, 0.3, 0.6, 0.3))
        batch.draw(shader)

        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        gpu.state.depth_test_set('NONE')

        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_plane})#
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
        batch.draw(shader)
        
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_arrow})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
        batch.draw(shader)
        gpu.state.depth_mask_set(False)

    # except:
    #     pass

class UserSettings():
    def __init__(self):
        self.GetSnappingSettings()
        self.GetUseDragImmediately()
        bpy.context.preferences.inputs.use_drag_immediately = True

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

    def GetUseDragImmediately(self):
        self.use_drag_immediately = bpy.context.preferences.inputs.use_drag_immediately

    def SetUseDragImmediately(self):
        bpy.context.preferences.inputs.use_drag_immediately = self.use_drag_immediately

    def ReturnAllSettings(self):
        self.SetUseDragImmediately()
        self.SetSnappingSettings()

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
        self.Toolname = ""
        self.header_text = ""
        self.SkipFrameValue = 2
        self.SkipFrameCount = 2
        self.OldMousePos = mathutils.Vector((0.0,0.0,0.0))
        self.NewMousePos = mathutils.Vector((0.0,0.0,0.0))
        self.Expected_Action = None
        """Function which be calling after delay"""
        self.PivotPoint = None
        self.TransfromOrientationMatrix = None
        self.NormalIntersectionPlane = None
        
        self.Event = bpy.types.Event
        """Temp Variable for saving event"""

        self.ActionsState = ActionsState()
        self.ORI = ORI()
        self.UserSettings = UserSettings()
        self.GenerateLambdaConditions()
        self.GenerateDelegates()

        self.GML = lambda event: GetMouseLocation(self.PivotPoint,self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
        """Just for convenience, to use the shorthand notation """
        #self.DrawCallBack_delegat = self.DrawCallBack # Empty drawcallback function

    def DrawCallBack(self, context):
        pass
    def GenerateLambdaConditions(self):
        """Conditions for action, can be overridden at __init__ at children classes"""
        self.If_Modify = lambda event: event.shift or event.alt
        self.If_Pass = lambda event: (self.If_MMove(event) and self.ActionsState.Pass) and self.SkipFrameCount > 0

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
        self.LM_D = lambda event: self.StartDelay(event, self.AfterLeftMouseAction, self.BedoreLeftMouseAction, True)
        # -----------------------RIGHT_MOUSE Exlude Axis-----------------------------------------------------#
        self.RM_D = lambda event: self.StartDelay(event, self.AfterRightMouseAction, self.BeforeRightMouseAction)
        # -----------------------MIDDLE_MOUSE No Constrain---------------------------------------------------#
        self.MM_D = lambda event: self.StartDelay(event, self.AfterMiddleMouseAction, self.BeforeMiddleMouseAction)
        # -----------------------MOVE_MOUSE No Constrain---------------------------------------------------#
        self.MoveM_D = lambda event: self.StartDelay(event, self.AfterMoveMouseAction, self.BeforeMoveMouseAction)
        # -----------------------SPACEBAR Bottom-------------------------------------------------------------#
        self.Space_D = lambda event: self.StartDelay(event, self.AfterSpaceAction, self.BeforeSpaceAction)
        #----------------------Tweak with new selection------------------------------------------------------#
        self.Shift_D = lambda event: self.StartDelay(event, self.AfterShiftAction , self.BeforeShiftAction, True)
        #----------------------Tweak with old selection------------------------------------------------------#
        self.Alt_D = lambda event : self.StartDelay(event, self.AfterAltAction , self.BeforeAltAction, True)

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
    def AfterShiftAction(self, event):
        self.ActionsState.Shift = False
        return self.ORI.RUNNING_MODAL
    def BeforeAltAction(self, event):
        self.OldMousePos = self.GML(event)
        self.ActionsState.Alt = True
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

    def SetUp(self, event):
        self.PivotPoint = GetPivotPointPoistion()
        self.TransfromOrientationMatrix = GetTransfromOrientationMatrix()
        self.NormalIntersectionPlane = GetNormalForIntersectionPlane(self.TransfromOrientationMatrix)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self.DrawCallBack, (context, ), 'WINDOW','POST_VIEW')# POST_PIXEL # POST_VIEW
        #self.OldMousePos = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
        #CalculatePointForStartDrawing(self)

    def Exit(self):
        print("Exit")
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
    def Cancel(self):
        self.Exit()
        return self.ORI.CANCELLED
    def Finish(self):
        self.Exit()
        return self.ORI.FINISHED

    def ModalReturnDec(self, value):
        if   value == self.ORI.RUNNING_MODAL: return self.ORI.RUNNING_MODAL
        elif value == self.ORI.PASS_THROUGH: return self.ORI.PASS_THROUGH
        elif value == self.ORI.INTERFACE: return self.ORI.INTERFACE
        elif value == self.ORI.FINISHED: return self.Finish()
        elif value == self.ORI.CANCELLED: return self.Cancel()

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "VIEW_3D" and len(context.selected_objects) > 0
    
    def StartDelay(self, event, action_after_delay, action_before_delay, use_delay = False):
        if use_delay:
            use_delay = self.SkipFrameCount
        self.TransfromOrientationMatrix = GetTransfromOrientationMatrix()
        self.PivotPoint = GetPivotPointPoistion()
        self.NormalIntersectionPlane = GetNormalForIntersectionPlane(self.TransfromOrientationMatrix)

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
        if self.If_Esc(event): return self.ModalReturnDec(self.Cancel())
        # -----------------------Skip Frames-------------------------------------------------------------#
        if self.If_Pass(event):
            self.SkipFrameCount -= 1
            return self.ORI.RUNNING_MODAL
        elif self.Expected_Action and self.ActionsState.Pass: 
            self.ActionsState.Pass = False
            self.NewMousePos = GetMouseLocation(self.PivotPoint, self.NormalIntersectionPlane, self.TransfromOrientationMatrix, event) 
            return self.ModalReturnDec(self.Expected_Action(self.Event))
        # -----------------------LEFT_MOUSE Only Axis Move---------------------------------------------------#
        if self.If_LM_Cond(event): return self.ModalReturnDec(self.LM_D(event))
        # -----------------------RIGHT_MOUSE Exlude Axis-----------------------------------------------------#
        if self.If_RM_Cond(event): return self.ModalReturnDec(self.RM_D(event))
        # -----------------------MIDDLE_MOUSE No Constrain---------------------------------------------------#
        if self.If_MM_Cond(event): return self.ModalReturnDec(self.MM_D(event))
        # -----------------------SPACEBAR Bottom-------------------------------------------------------------#
        if self.If_Spcae_Cond(event): return self.ModalReturnDec(self.Space_D(event))
        #----------------------Tweak with new selection------------------------------------------------------#
        if self.If_Shift_Cond(event): return self.ModalReturnDec(self.Shift_D(event))
        #----------------------Tweak with old selection------------------------------------------------------#
        if self.If_Alt_Cond(event): return self.ModalReturnDec(self.Alt_D(event))
        # -----------------------MOVE_MOUSE------------------------------------------------------------------#
        if self.If_MMove_Cond(event): return self.ModalReturnDec(self.MoveM_D(event))
        if self.If_G(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.G, self.After_G, self.Before_G, 0))
        if self.If_X(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.X, self.After_X, self.Before_X, 0))
        if self.If_Y(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.Y, self.After_Y, self.Before_Y, 0))
        if self.If_Z(event):
            return self.ModalReturnDec(self.StartDelay(event, self.ActionsState.Z, self.After_Z, self.Before_Z, 0))
        
        return self.ORI.RUNNING_MODAL
    
    def invoke(self, context, event):
        context.area.header_text_set(self.header_text)       
        if context.space_data.type == 'VIEW_3D':
            self.SetUp(event)
            context.window_manager.modal_handler_add(self)
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
        self.If_Alt_Cond =   lambda event:   self.If_Alt(event) and (self.If_LM(event) or self.If_RM(event) or self.If_MM(event))
        self.If_Shift_Cond = lambda event: self.If_Shift(event) and (self.If_LM(event) or self.If_RM(event) or self.If_MM(event))

    def AfterLeftMouseAction(self, event):
        axis = GetMouseDirection(self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        SetConstarin.SetMoveOnlyOneAxis(axis)
        return self.ORI.FINISHED
    def AfterRightMouseAction(self, event):
        SetConstarin.SetMoveExclude(GetAxixForTransformation(self.NormalIntersectionPlane))
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
    def AfterShiftAction(self, event):
        if context.mode == "EDIT_MESH":
            bpy.ops.mesh.select_all(action='DESELECT')
        else:
            bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.view3d.select('INVOKE_DEFAULT', extend=True, deselect=False, enumerate=False, toggle=False)
        if self.If_LM(event):
            self.AfterLeftMouseAction(event)
        if self.If_RM(event):
            self.AfterRightMouseAction(event)
        if self.If_MM(event):
            self.AfterMiddleMouseAction(event)
        self.ActionsState.Shift = False
        self.Expected_Action = None
        return self.ORI.FINISHED
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
        self.toolName = "Advanced Scale"
        self.If_Alt_Cond =   lambda event:   self.If_Alt(event) and (self.If_LM(event) or self.If_RM(event) or self.If_MM(event))
        self.If_Shift_Cond = lambda event: self.If_Shift(event)

    def AfterLeftMouseAction(self, event):
        axis = GetMouseDirection(self.OldMousePos, self.NewMousePos, self.TransfromOrientationMatrix)
        SetConstarin.SetScaleOnly(axis)
        return self.ORI.FINISHED
    def BeforeRightMouseAction(self, event):
        SetConstarin.SetScaleExclude(GetAxixForTransformation(self.NormalIntersectionPlane))
        return self.ORI.FINISHED
    
    def AfterSpaceAction(self, event):
        bpy.ops.view3d.advancedscale_zero('INVOKE_DEFAULT')
        
        return {'FINISHED'}
    
    def AfterMiddleMouseAction(self, event):
        SetConstarin.SetScaleNoConstrain()
        return self.ORI.FINISHED
    
    def AfterShiftAction(self, event):
        deb("scale")
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

class AdvancedScaleZore(AdvancedTransform):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale_zero"
    bl_label = "Advanced Scale zero"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self) -> None:
        super().__init__()
        self.MirrorAxis = Vector((0.0,0.0,0.0))
        self.If_MMove_Cond = lambda event: self.If_MMove(event)
        

        
    def AfterSpaceAction(self, event):
        if event.value == "RELEASE":
            SetConstarin.SetScaleOnlySetZero(self.MirrorAxis)
            return self.ORI.FINISHED
        else:
            self.ORI.PASS_THROUGH
    
    def AfterMoveMouseAction(self, event):
        self.NewMousePos = self.GML(event)
        mouse_direction = (self.NewMousePos - self.PivotPoint).normalized()
        self.MirrorAxis = GetAxixForTransformation(self.NewMousePos)
        deb(self.MirrorAxis, "self.MirrorAxis", self.NewMousePos, (self.NewMousePos - self.PivotPoint).normalized())
        return self.ORI.PASS_THROUGH

class AdvancedScaleMirror(AdvancedTransform):
    ''' Advanced Scale '''
    bl_idname = "view3d.advancedscale_mirror"
    bl_label = "Advanced Scale Mirror"
    bl_options = {'REGISTER', 'UNDO'}
    def __init__(self) -> None:
        super().__init__()
        self.MirrorAxis = Vector((0.0,0.0,0.0))
        self.If_MMove_Cond = lambda event: self.If_MMove(event)
        self.If_Shift_Cond = lambda event: (event.type == 'LEFT_SHIFT' and event.value == "RELEASE")
        #self.DrawCallBack = lambda self, context: None
        #self.DrawCallBack = self.DrawCallBack
        self.DrawCallBack_delegat = self.DrawCallBack#partial(self.DrawCallBack)

    def DrawCallBack(self, context):
    #try:
        if self.TransfromOrientationMatrix != None and self.MirrorAxis != None and self.PivotPoint != None:
            # Get Data for rotation
            direction_axis_index = 0 if self.MirrorAxis == 'x' else (1 if self.MirrorAxis == 'y' else 2)
            direction_vector = self.TransfromOrientationMatrix[direction_axis_index].to_3d() 

            current_direction = V((+0.0, +1.0, 0.0)) if direction_axis_index == 1 else V((+0.0, +1.0, 0.0))*-1 # i don't know why but axies X and Z inverted and i must flip them
            angle = current_direction.angle(direction_vector) # get angle between original direction and desired rirection
            axis_for_rotation = current_direction.cross(direction_vector) # Axis for rotation

            # Rotate Vector By Transform Orientation Matrix
            RTM = lambda vector: (vector @ mathutils.Matrix.Rotation(angle, 4, axis_for_rotation)) + self.PivotPoint
            offset = V((0,3,0))
            # arrow vertices position with offset and alinge by matrix
            a_v1 = RTM(V((+0.0, +1.0, +0.0)) + offset )
            a_v2 = RTM(V((-1.0, +0.0, +0.0)) + offset )
            a_v3 = RTM(V((-0.4, +0.0, +0.0)) + offset )
            a_v4 = RTM(V((-0.4, -2.0, +0.0)) + offset )
            a_v5 = RTM(V((+0.4, -2.0, +0.0)) + offset )
            a_v6 = RTM(V((+0.4, +0.0, +0.0)) + offset )
            a_v7 = RTM(V((+1.0, +0.0, +0.0)) + offset )

            # plane
            p_v1 = RTM(V((+6.0, +0.0, +6.0)))
            p_v2 = RTM(V((+6.0, +0.0, -6.0)))
            p_v3 = RTM(V((-6.0, +0.0, -6.0)))
            p_v4 = RTM(V((-6.0, +0.0, +6.0)))

            #Grid lines
            g_v1  = RTM(V((-4.0000, -0.0000, 6.0000)))
            g_v2  = RTM(V((-4.0000, 0.0000, -6.0000)))
            g_v3  = RTM(V((-2.0000, -0.0000, 6.0000)))
            g_v4  = RTM(V((-2.0000, 0.0000, -6.0000)))
            g_v5  = RTM(V((+0.0000, -0.0000, 6.0000)))
            g_v6  = RTM(V((+0.0000, 0.0000, -6.0000)))
            g_v7  = RTM(V((+2.0000, -0.0000, 6.0000)))
            g_v8  = RTM(V((+2.0000, 0.0000, -6.0000)))
            g_v9  = RTM(V((+4.0000, -0.0000, 6.0000)))
            g_v10 = RTM(V((+4.0000, 0.0000, -6.0000)))
            g_v11 = RTM(V((-6.0000, -0.0000, 4.0000)))
            g_v12 = RTM(V((+6.0000, -0.0000, 4.0000)))
            g_v13 = RTM(V((-6.0000, -0.0000, 2.0000)))
            g_v14 = RTM(V((+6.0000, -0.0000, 2.0000)))
            g_v15 = RTM(V((-6.0000, 0.0000, +0.0000)))
            g_v16 = RTM(V((+6.0000, 0.0000, +0.0000)))
            g_v17 = RTM(V((-6.0000, 0.0000, -2.0000)))
            g_v18 = RTM(V((+6.0000, 0.0000, -2.0000)))
            g_v19 = RTM(V((-6.0000, 0.0000, -4.0000)))
            g_v20 = RTM(V((+6.0000, 0.0000, -4.0000)))


            # make triangles, each triange shuld veve 3 vertices. that method not use idexes only vertices order
            arrow_vertices =[a_v1, a_v2, a_v3,
                            a_v3, a_v4, a_v5,
                            a_v3, a_v5, a_v6,
                            a_v1, a_v6, a_v7,
                            a_v1, a_v3, a_v6]
            
            plane =         [p_v1, p_v2, p_v4, 
                            p_v2 ,p_v4, p_v3]
            # make loop for conture
            contur_arrow = [a_v1, a_v2, a_v3, a_v4, a_v5, a_v6, a_v7]
            contur_plane = [p_v1,p_v2,p_v3,p_v4]
            grid = [g_v1,g_v2,g_v3,g_v4,g_v5,g_v6,g_v7,g_v8,g_v9,g_v10,g_v11,g_v12,g_v13,g_v14,g_v15,g_v16,g_v17,g_v18,g_v19,g_v20]

            # Facing 
            # ApplyFacing = lambda angle, dir, arr: [(i) @ mathutils.Matrix.Rotation(angle, 4, dir) for i in arr]
            # SignedAngle= lambda v1, v2, axis: v1.angle(v2) * (1 if axis.dot(v1.cross(v2)) >= 0 else -1)

            # direction_vector_camera = (bpy.context.region_data.view_rotation @ Vector((0, 0, -1))) # get view view direction
            # up_direction = axis_for_rotation.cross(direction_vector)
            # angle_to_camera = SignedAngle(up_direction, direction_vector_camera, direction_vector) * -1

            # arrow_vertices = ApplyFacing(angle_to_camera, direction_vector, arrow_vertices)
            # contur_arrow = ApplyFacing(angle_to_camera, direction_vector, contur_arrow)

            # Make shader and batsh
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            gpu.state.blend_set("ALPHA")
            gpu.state.line_width_set(3)

            batch = batch_for_shader(shader, 'TRIS', {"pos": plane})
            shader.uniform_float("color", (0.0, 0.3, 0.6, 0.3))
            batch.draw(shader)

            batch = batch_for_shader(shader, 'TRIS', {"pos": arrow_vertices})
            shader.uniform_float("color", (0.0, 0.3, 0.6, 0.3))
            batch.draw(shader)

            shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
            gpu.state.depth_test_set('NONE')


            batch = batch_for_shader(shader, 'LINES', {"pos": grid})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.15))
            batch.draw(shader)

            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_plane})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
            batch.draw(shader)
            
            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": contur_arrow})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))
            batch.draw(shader)
            gpu.state.depth_mask_set(False)

    def AfterShiftAction(self, event):
        SetConstarin.SetScaleMirror(self.MirrorAxis)
        return self.ORI.FINISHED
    
    def AfterMoveMouseAction(self, event):
        self.NewMousePos = self.GML(event)
        mouse_direction = (self.NewMousePos - self.PivotPoint).normalized()
        self.MirrorAxis = GetAxixForTransformation(self.NewMousePos)
        self.report({'INFO'},"") # Update modal and check shift event. Remove bag with wating new tick
        return self.ORI.PASS_THROUGH

class AdvancedRotation(AdvancedTransform):
    """ Advanced move """
    bl_idname = "view3d.advanced_rotation"
    bl_label = "Advanced Rotation"
    #bl_options = {'REGISTER', 'UNDO'}

    def __init__(self):
        super().__init__()
        self.header_text = 'LMB constraint view axis, RMB constraint view axis snap, MMB free rotate'
        self.ToolName = 'Advanced Rotation'
        self.RotationValue = 0.0

        self.LM_D = lambda event: self.StartDelay(event, self.AfterLeftMouseAction, self.BedoreLeftMouseAction)
        self.If_LM_Cond = lambda event: self.If_LM(event)# and (self.ActionsState.LeftMouse == False and self.ActionsState.MoveMouse == False)
        self.If_RM_Cond = lambda event: self.If_RM(event) and (self.ActionsState.MoveMouse == False and self.ActionsState.LeftMouse == False)
        self.If_MMove_Cond = lambda event: self.If_MMove(event) and (self.ActionsState.LeftMouse or self.ActionsState.RightMouse)
        self.AngleSnappingStep = int(context.preferences.addons[__name__].preferences.Snapping_Step)

    def GetAngle(self, v1, v2):
        v1 = v1 - self.PivotPoint
        v2 = v2 - self.PivotPoint
        # we can get "ValueError: math domain error" or ZeroDivisionError:
        try:
            cos_angle = v1.dot(v2) / (v1.length * v2.length)
            angle = math.acos(cos_angle)
            return math.degrees(angle)
        except:
            print("ERRRRRROOOOORR")
            return None
        
    def AfterLeftMouseAction(self, event):
        if event.value == 'PRESS':
            self.ActionsState.MoveMouse = True
            return self.ORI.RUNNING_MODAL
        elif event.value == 'RELEASE':
            bl_options = {'REGISTER', 'UNDO'}
            return self.ORI.FINISHED

    def AfterRightMouseAction(self, event):
        axis = GetAxixForTransformation(self.NormalIntersectionPlane)
        SetConstarin.SetRotationOnly(axis)
        return self.ORI.FINISHED

    def BeforeMoveMouseAction(self, event):
        pass
    
    def AfterMoveMouseAction(self, event):
        self.NewMousePos = self.GML(event)
        angle = self.GetAngle(self.OldMousePos, self.NewMousePos)

        # Check rotation step
        if angle != None and round(angle / self.AngleSnappingStep) * self.AngleSnappingStep != 0:
            angle = self.AngleSnappingStep
            print("rot")
            # find third axis 1 is mouse direction 2 is view direction  and 3 (corss) look at pivot point
            cross=((self.NewMousePos - self.OldMousePos).normalized()).cross(self.NormalIntersectionPlane)
            # if value biger then 0 counterclock-wise else clockwise
            angle = angle*-1 if cross.dot(self.NewMousePos - self.PivotPoint) > 0 else angle

            self.OldMousePos = self.NewMousePos.copy()
            self.RotationValue += angle
            
            SetConstarin.SetRotationOnlyAT(angle , GetAxixForTransformation(self.NormalIntersectionPlane))
        return self.ORI.RUNNING_MODAL

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
        return (axis == 'x', axis == 'y', axis == 'z')
    @staticmethod
    def SetupExcludeAxis(axis):
        return (not (axis == 'x'), not (axis == 'y'), not (axis == 'z'))

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
        bpy.ops.transform.rotate(value=math.radians(value),orient_axis=axis.upper(), constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetRotationOnly(axis):
        bpy.ops.transform.rotate('INVOKE_DEFAULT', orient_axis=axis.upper(), constraint_axis=SetConstarin.SetupSingleAxis(axis))
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
        axis_value = {'x': (0.0, 1.0, 1.0), 'y': (1.0, 0.0, 1.0), 'z': (1.0, 1.0, 0.0)}
        bpy.ops.transform.resize(value=axis_value[axis], constraint_axis=SetConstarin.SetupSingleAxis(axis))
    @staticmethod
    def SetScaleExcludeSetZero(axis):
        axis_value = {'x': (0.0, 1.0, 1.0), 'y': (1.0, 0.0, 1.0), 'z': (1.0, 1.0, 0.0)}
        bpy.ops.transform.resize(value=axis_value[axis], constraint_axis=SetConstarin.SetupExcludeAxis[axis])
    @staticmethod
    def SetScaleMirror(axis):
        axis_value = {'x': (-1.0, 1.0, 1.0), 'y': (1.0, -1.0, 1.0), 'z': (1.0, 1.0, -1.0)}
        bpy.ops.transform.resize(value=(-1.0, 1.0, 1.0), constraint_axis=SetConstarin.SetupSingleAxis(axis))
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
