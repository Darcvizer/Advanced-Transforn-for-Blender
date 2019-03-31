from bpy.types import Operator
import bpy
from bpy import context as C
from mathutils import Vector as V
from mathutils import Matrix
from bpy import context
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from bgl import glLineWidth, glEnable, GL_BLEND, glDisable
from bpy.types import Header
import bpy
import bgl
import blf
import bmesh
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix
from bpy.props import IntProperty
from bpy.types import Operator, Macro
from bpy.props import IntProperty, FloatProperty
from bpy.utils import register_class, unregister_class
from mathutils.geometry import intersect_line_plane, intersect_point_quad_2d, intersect_line_line_2d
import numpy as np
import math
import os
import gpu
from gpu_extras.batch import batch_for_shader
import bgl
import blf
import mathutils
import rna_keymap_ui
from bpy.props import (
	EnumProperty, BoolProperty )

bl_info = {
	"name": "Advanced Transform",
	"location": "View3D > Advanced Transform",
	"description": "Advanced Transform",
	"author": "Vladislav Kindushov(Darcvizer)",
	"version": (0, 3),
	"blender": (2, 80, 0),
	"category": "View3D",}

def GetView(self):
	# ______________________Get Camera Direction________________#
	direction = (self.mat.inverted() @ V(bpy.context.region_data.view_rotation @ V((0.0, 0.0, -1.0))))
	x = direction[0]
	y = direction[1]
	z = direction[2]
	# ______________________Set Coords Fro Plane________________#
	if abs(x) > abs(y) and abs(x) > abs(z):
		self.exc_axis = 'x'
		if x < 0:
			self.negativMlt = True
			self.V_matrix = Matrix.Translation(self.center) @ (
				Matrix(self.mat.to_3x3() @ Matrix.Rotation(-1.570796, 3, Vector((0.0, 1.0, 0.0)))).to_4x4())
		else:
			self.V_matrix = Matrix.Translation(self.center) @ (
					Matrix(self.mat.to_3x3() @ Matrix.Rotation(1.570796, 3, Vector((0.0, 1.0, 0.0)))).to_4x4())
	elif abs(y) > abs(x) and abs(y) > abs(z):
		self.exc_axis = 'y'
		if y < 0:
			self.negativMlt = True
			self.V_matrix = Matrix.Translation(self.center) @ (
				Matrix(self.mat.to_3x3() @ Matrix.Rotation(1.570796, 3, Vector((1.0, 0.0, 0.0)))).to_4x4())
		else:
			self.V_matrix = Matrix.Translation(self.center) @ (
				Matrix(self.mat.to_3x3() @ Matrix.Rotation(-1.570796, 3, Vector((1.0, 0.0, 0.0)))).to_4x4())

	elif abs(z) > abs(x) and abs(z) > abs(y):
		self.exc_axis = 'z'
		# print("AXIS _ ", z)
		if z < 0:
			self.negativMlt = True
			self.V_matrix = Matrix.Translation(self.center) @ (
				Matrix(self.mat.to_3x3() @ Matrix.Rotation(-1.570796*2, 3, Vector((1.0, 0.0, 0.0)))).to_4x4())
		else:
			self.V_matrix = Matrix.Translation(self.center) @ (
				Matrix(self.mat.to_3x3() @ Matrix.Rotation(1.570796*2, 3, Vector((0.0, 0.0, 0.0)))).to_4x4())

def GetCenter(self):
	###---------------------------------------Get Center Selection--------------------------------------------------###
	###-------------BOUNDING_BOX_CENTER----------------###
	if bpy.context.scene.tool_settings.transform_pivot_point == 'BOUNDING_BOX_CENTER':
		context.area.tag_redraw()
		self.oldPositionCursor = bpy.context.scene.cursor.location.copy()
		bpy.ops.view3d.snap_cursor_to_selected()
		self.center = bpy.context.scene.cursor.location.copy()
		bpy.context.scene.cursor.location = self.oldPositionCursor

	###--------------------CURSOR----------------------###
	elif bpy.context.scene.tool_settings.transform_pivot_point == 'CURSOR':
		self.center = bpy.context.scene.cursor.location.copy()

	###--------------------INDIVIDUAL_ORIGINS OR MEDIAN_POINT----------------------###
	elif bpy.context.scene.tool_settings.transform_pivot_point == 'INDIVIDUAL_ORIGINS' or bpy.context.scene.tool_settings.transform_pivot_point == 'MEDIAN_POINT':

		self.oldPositionCursor = bpy.context.scene.cursor.location.copy()
		bpy.ops.view3d.snap_cursor_to_selected()
		self.center = bpy.context.scene.cursor.location.copy()
		bpy.context.scene.cursor.location = self.oldPositionCursor.copy()

	###--------------------ACTIVE_ELEMENT----------------------###
	elif bpy.context.scene.tool_settings.transform_pivot_point == 'ACTIVE_ELEMENT':
		self.oldPositionCursor = bpy.context.scene.cursor.location.copy()
		bpy.ops.view3d.snap_cursor_to_active()
		self.center = bpy.context.scene.cursor.location.copy()
		bpy.context.scene.cursor.location = self.oldPositionCursor

def GetDirection(self):
	###-----------------------------------------------------Get Direction------------------------------------------------------###
	if bpy.context.scene.transform_orientation_slots[0].type == 'GLOBAL':
		self.mat = Matrix()
	elif bpy.context.scene.transform_orientation_slots[0].type == 'LOCAL':
		self.mat = bpy.context.active_object.rotation_euler.to_matrix()

	elif  bpy.context.scene.transform_orientation_slots[0].type == 'NORMAL':
		if bpy.context.mode == 'OBJECT':
			self.mat = bpy.context.active_object.rotation_euler.to_matrix()
		elif bpy.context.mode == 'EDIT_MESH':
			tempOrientTrans = bpy.context.scene.transform_orientation_slots[0].type
			bpy.ops.transform.create_orientation(name="AdvancedTransform", use_view=False, use=True,
												 overwrite=True)
			if bpy.context.scene.transform_orientation_slots[0].type == "AdvancedTransform":
				self.mat = bpy.context.scene.transform_orientation_slots[0].custom_orientation.matrix
				bpy.context.scene.transform_orientation_slots[0].type = tempOrientTrans

	elif bpy.context.scene.transform_orientation_slots[0].type == 'GIMBAL':
		self.mat = Matrix()

	elif bpy.context.scene.transform_orientation_slots[0].type == 'VIEW':
		tempOrientTrans = bpy.context.scene.transform_orientation_slots[0].type
		bpy.ops.transform.create_orientation(name="AdvancedTransform", use_view=True, use=True, overwrite=True)
		if bpy.context.scene.transform_orientation_slots[0].type == "AdvancedTransform":
			self.mat = bpy.context.scene.transform_orientation_slots[0].custom_orientation.matrix
			bpy.context.scene.transform_orientation_slots[0].type = tempOrientTrans

	elif bpy.context.scene.transform_orientation_slots[0].type == 'CURSOR':
		self.mat = bpy.context.scene.cursor.rotation_euler.to_matrix()


	else:
		self.mat = bpy.context.scene.transform_orientation_slots[0].custom_orientation.matrix

def UserPresets(self, context, Set=False):
	if Set is False:
		self.save_user_drag = context.preferences.inputs.use_drag_immediately
		if self.save_user_drag is not True:
			context.preferences.inputs.use_drag_immediately = True

	elif Set is True:
		context.preferences.inputs.use_drag_immediately = self.save_user_drag

def UserSnap(self, context, Set=False):
	if Set == False:
		self.user_snap_element = context.scene.tool_settings.snap_elements
		self.user_snap_target = context.scene.tool_settings.snap_target
		self.user_snap_rot = context.scene.tool_settings.use_snap_align_rotation
		self.user_snap_project = context.scene.tool_settings.use_snap_project
		self.user_snap = context.scene.tool_settings.use_snap
	elif Set == True:
		context.scene.tool_settings.snap_elements = self.user_snap_element
		context.scene.tool_settings.snap_target = self.user_snap_target
		context.scene.tool_settings.use_snap_align_rotation = self.user_snap_rot
		context.scene.tool_settings.use_snap_project = self.user_snap_project
		context.scene.tool_settings.use_snap = self.user_snap

def SnapMoveOrientation(self, context):
	context.scene.tool_settings.snap_elements = {'FACE'}
	context.scene.tool_settings.snap_target = 'CENTER'
	context.scene.tool_settings.use_snap_align_rotation = True
	context.scene.tool_settings.use_snap_project = True
	context.scene.tool_settings.use_snap = True

def GetCoordMouse(self, context, event, point=None, matrix=None, revers=False):
	"""
	convert mouse pos to 3d point over plane defined by origin and normal
	"""
	point = self.center
	matrix = self.V_matrix
	# get the context arguments
	region = bpy.context.region
	rv3d = bpy.context.region_data
	coord = event.mouse_region_x, event.mouse_region_y
	view_vector_mouse = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
	ray_origin_mouse = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
	loc = intersect_line_plane(ray_origin_mouse, ray_origin_mouse + view_vector_mouse, point,
							   matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)), False)
	return loc

def SetupAxis(self, temp_loc_first, temp_loc_last):
	vec = (self.mat.to_3x3().inverted() @ (temp_loc_last - temp_loc_first).normalized())

	if abs(vec[0]) > abs(vec[1]) and abs(vec[0]) > abs(vec[2]):
		return 'x'
	elif abs(vec[1]) > abs(vec[0]) and abs(vec[1]) > abs(vec[2]):
		return 'y'
	elif abs(vec[2]) > abs(vec[0]) and abs(vec[2]) > abs(vec[1]):
		return 'z'

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

	#-----Get full circle 3d-----#
	for i in range(1, 361):
		angle = 2 * math.pi * i / 360
		x = 5 * math.cos(angle)
		y = 5 * math.sin(angle)
		self.MailCircle_3d.append(V((x, y, 0)))
		angle = 2 * math.pi * (i - 1) / 360
		x = 5 * math.cos(angle)
		y = 5 * math.sin(angle)
		self.MailCircle_3d.append(V((x, y, 0)))

		#--------To 2d-------#
	step = 0
	for i in self.MailCircle_3d:
			step += 1
			self.MainCirclePositiv.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															   (self.RotaionMatrix @ MS) @ c(i,-0.1)))
			self.MainCircleNegativ.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															   (self.NegativeMatrix @ MS) @ c(i,-0.1)))
			if step == 2:
				self.MainCirclePositiv.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
														  (self.RotaionMatrix @ MS) @ V((0, 0, 0))))
				self.MainCircleNegativ.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
														  (self.NegativeMatrix @ MS) @ V((0, 0, 0))))
				step = 0

	#--------get the second circle---------#
	offset = 0.9
	for i in range(0, len(self.MailCircle_3d)-2):
		self.SecondPoint_3d.append(self.MailCircle_3d[i])
		self.SecondPoint_3d.append(self.MailCircle_3d[i + 1])
		self.SecondPoint_3d.append(V((self.MailCircle_3d[i][0] * offset, self.MailCircle_3d[i][1] * offset, self.MailCircle_3d[i][2] * offset)))  # + coords[i]
		self.SecondPoint_3d.append(V((self.MailCircle_3d[i][0] * offset, self.MailCircle_3d[i][1] * offset, self.MailCircle_3d[i][2] * offset)))  # + coords[i]
		self.SecondPoint_3d.append(V((self.MailCircle_3d[i + 1][0] * offset, self.MailCircle_3d[i + 1][1] * offset, self.MailCircle_3d[i + 1][2] * offset)))  # + coords[i+1]
		self.SecondPoint_3d.append(self.MailCircle_3d[i + 1])
		# --------To 2d-------#
	for i in self.SecondPoint_3d:
		self.SecondCirclePositiv.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															   (self.RotaionMatrix @ MS) @ i))
		self.SecondCircleNegativ.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															   (self.NegativeMatrix @ MS) @ i))

	#-----Get line
	flip_flop = True
	offsetLine = 0.1
	offset = 0.5
	step = 0
	for i in range(0,len(self.MailCircle_3d)-1,20):# 0# 20 #+1
			self.line.append(c(self.MailCircle_3d[i+1],-0.025))
			self.line.append(c(self.MailCircle_3d[i+1], -0.075))


	for i in range(10,len(self.MailCircle_3d)-1,20):#10#20#+1
			self.line.append(c(self.MailCircle_3d[i+1],0))
			self.line.append(c(self.MailCircle_3d[i+1],-0.1))



	for i in self.line:
		self.LongLine_2d.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
													  (self.RotaionMatrix @ MS) @ i))
	if int(context.preferences.addons[__name__].preferences.Snapping_Step) != 1:
		for i in range(1, len(self.MailCircle_3d)-1, self.Step*2):
			self.SnapLine.append(c(self.MailCircle_3d[i],-0.1))
			self.SnapLine.append(V((0,0,0)))

	for i in self.SnapLine:
		self.SnapLine_2d.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
													  (self.RotaionMatrix @ MS) @ i))


	for i in range(0, len(self.MailCircle_3d)-1):
		self.edging1.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
													  (self.RotaionMatrix @ MS) @ self.MailCircle_3d[i]))
		self.edging2.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
													  (self.RotaionMatrix @ MS) @ (c(self.MailCircle_3d[i],-0.1))))

	self.firstPoint = (view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
												  (self.RotaionMatrix @ MS) @ V((0,0,0))))
def f (self, context, event):
	# print("self.Angle", self.Angle)
	if self.call_no_snap == 20:
		self.mousePosition = event.mouse_region_x, event.mouse_region_y
		self.temp_loc_last = GetCoordMouse(self, context, event)
		self.Angle = int(round(
			GetAngle(self, self.temp_loc_last, self.temp_loc_first, self.V_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)))))
		self.realAngle = self.Angle
		intersect_Quad(self, event)
		self.call_no_snap = 0
	else:
		self.call_no_snap += 1

def DrawCalback(self, context, event):
	try:

		#------------------Draw 2D Quad-----------------#
		# glEnable(GL_BLEND)
		# coord = [self.LD, self.CD, self.C, self.LC]
		#
		# indices = ((0, 1, 2), (2, 3, 0))
		# shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		# batch = batch_for_shader(shader, 'TRIS', {"pos": coord}, indices=indices)
		# shader.bind()
		# shader.uniform_float("color", (1.0, 0.0, 0.0, 0.25))
		# batch.draw(shader)
		# glDisable(GL_BLEND)
		#
		# glEnable(GL_BLEND)
		# coord = [self.CD, self.RD, self.RC, self.C]
		# indices = ((0, 1, 2), (2, 3, 0))
		# shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		# batch = batch_for_shader(shader, 'TRIS', {"pos": coord}, indices=indices)
		#
		# shader.bind()
		# shader.uniform_float("color", (0.0, 1.0, 0.0, 0.25))
		# batch.draw(shader)
		# glDisable(GL_BLEND)
		#
		# glEnable(GL_BLEND)
		# coord = [self.C, self.RC, self.RU, self.CU]
		# indices = ((0, 1, 2), (2, 3, 0))
		# shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		# batch = batch_for_shader(shader, 'TRIS', {"pos": coord}, indices=indices)
		#
		# shader.bind()
		# shader.uniform_float("color", (0.0, 0.0, 1.0, 0.25))
		# batch.draw(shader)
		# glDisable(GL_BLEND)
		#
		# glEnable(GL_BLEND)
		# coord = [self.LC, self.C, self.CU, self.LU]
		# indices = ((0, 1, 2), (2, 3, 0))
		# shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		# batch = batch_for_shader(shader, 'TRIS', {"pos": coord}, indices=indices)
		#
		# shader.bind()
		# shader.uniform_float("color", (1.0, 1.0, 1.0, 0.25))
		# batch.draw(shader)
		# glDisable(GL_BLEND)
		if self.is_snap == False:
			f(self, context, event)


		glEnable(GL_BLEND)
		# =----------------Draw main circle---------------------#
		if self.ModeRotation:
			shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
			batch = batch_for_shader(shader, 'TRIS', {"pos": self.MainCircleNegativ[:(self.realAngle * 3) * -1]})
			shader.bind()
			shader.uniform_float("color", self.color)
			batch.draw(shader)
		else:
			shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
			batch = batch_for_shader(shader, 'TRIS', {"pos": self.MainCirclePositiv[:self.realAngle * 3]})
			shader.bind()
			shader.uniform_float("color", self.color)
			batch.draw(shader)


		if self.ModeRotation:
			shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
			batch = batch_for_shader(shader, 'TRIS', {"pos": self.SecondCircleNegativ[(self.Angle * 12) * -1:]})
			shader.bind()
			shader.uniform_float("color", (0.5, 0.5, 0.5, 0.1))
			batch.draw(shader)
		else:
			shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
			batch = batch_for_shader(shader, 'TRIS', {"pos": self.SecondCirclePositiv[self.Angle * 12:]})
			shader.bind()
			shader.uniform_float("color", (0.5, 0.5, 0.5, 0.1))
			batch.draw(shader)


		if self.ModeRotation:
			shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
			batch = batch_for_shader(shader, 'TRIS', {"pos": self.SecondCircleNegativ[:(self.Angle * 12) * -1]})
			shader.bind()
			shader.uniform_float("color", (1, 0.4, 0.2, 0.2))
			batch.draw(shader)
		else:
			shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
			batch = batch_for_shader(shader, 'TRIS', {"pos": self.SecondCirclePositiv[:self.Angle * 12]})
			shader.bind()
			shader.uniform_float("color", (1, 0.4, 0.2, 0.2))
			batch.draw(shader)

		glLineWidth(2)
		shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		batch = batch_for_shader(shader, 'LINES', {"pos": self.LongLine_2d})
		shader.bind()
		shader.uniform_float("color", (0, 0, 0, 0.5))
		batch.draw(shader)

		#glLineWidth(3)
		shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		batch = batch_for_shader(shader, 'LINES', {"pos": self.SnapLine_2d})
		shader.bind()
		shader.uniform_float("color", (1.0, 1.0, 1.0, 0.05))
		batch.draw(shader)

		#glLineWidth(3)
		shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		batch = batch_for_shader(shader, 'LINES', {"pos": self.edging1})
		shader.bind()
		shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
		batch.draw(shader)

		# glLineWidth(3)
		shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		batch = batch_for_shader(shader, 'LINES', {"pos": self.edging2})
		shader.bind()
		shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
		batch.draw(shader)

		glLineWidth(3)
		line = []
		line.append(self.firstPoint)
		line.append(self.edging2[0])
		line.append(self.edging2[self.realAngle*2])
		line.append(self.firstPoint)

		shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		batch = batch_for_shader(shader, 'LINES', {"pos": line})
		shader.bind()
		shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
		batch.draw(shader)

		blf.position(0, self.mousePosition[0], self.mousePosition[1], 0)
		blf.size(0, 22, 40)
		blf.draw(0, str(round(self.Angle)))

		glDisable(GL_BLEND)
	except:
		pass

def GetAngle(self, v2, v1, n, radian = False):
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
			return math.degrees(math.atan2(det * -1, dot * -1))  + self.addAngle
		elif self.addAngle == 0:
			return math.degrees(math.atan2(det, dot))# + self.addAngle

def StartDraw(self, context, event):
	'''Setup setings for rotation'''
	#-----------------Rotation Matrix on directio the first click------------------#
	radians = GetAngle(self,(self.V_matrix @ Vector((0.0, 1.0, 0.0))) ,self.temp_loc_first,
						  self.V_matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)), radian = True)
	# print("radians - ", radians)
	axis_dst = Vector((0.0, 1.0, 0.0))
	self.directionMatrix = (self.temp_loc_first).normalized()
	#matrix_rotate = Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0))).to_3x3() @ (axis_dst.rotation_difference(self.V_matrix.to_3x3().inverted() @ ((self.directionMatrix)).normalized()).to_matrix().to_3x3())
	matrix_rotate = Matrix.Rotation(radians+-1.570796, 3, Vector((0.0, 0.0, -1.0))).to_4x4()
	self.RotaionMatrix = Matrix.Translation(self.center) @ (self.V_matrix.to_3x3() @ matrix_rotate.to_3x3()).to_4x4()
	NM = Matrix.Rotation(-1.570796*2, 3, Vector((1.0, 0.0, 0.0))).to_3x3()
	self.NegativeMatrix = Matrix.Translation(self.center) @ (self.RotaionMatrix.to_3x3() @ NM.to_3x3()).to_4x4()
	#Matrix.Translation(self.center) @
	# -------------------Get circle size---------------------#
	for i in bpy.context.window.screen.areas:
		if i.type == 'VIEW_3D':
			CD = i.spaces[0].region_3d.view_distance
			MS = Matrix.Scale(1, 4)


	def Rot(self, x, y):
		"""Rotation 2d quad on the first click"""
		#v1 = V((C.region.width/2,C.region.height /2 ))
		v1 = view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)
		v2 = view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.temp_loc_first)
		radians = math.atan2(v1[1]-v2[1], v1[0]-v2[0]) * -1

		# print("angle", math.degrees(radians))
		# print("v1",v1)
		# print("v2",v2)
		# print("radians",radians)
		#origin = V((C.region.width/2,C.region.height /2 ))
		origin = view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)
		centerOffset = origin - V((bpy.context.region.width/2,bpy.context.region.height /2 ))
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
		return V((qx,qy))

	# Высота C.region.height
	# Ширена C.region.width
	zero = (bpy.context.region.width * 2)
	max = (bpy.context.region.width * 2) * -1

	self.LD = Rot(self, V((zero,zero))[0], V((zero,zero))[1])
	self.LC = Rot(self, V((zero,zero))[0], V((zero,bpy.context.region.height/2))[1])
	self.LU = Rot(self, V((zero,bpy.context.region.height + max))[0], V((zero,bpy.context.region.height + max))[1])
	self.RD = Rot(self, V((bpy.context.region.width + max,zero))[0], V((bpy.context.region.width + max,zero))[1])
	self.RC = Rot(self, V((bpy.context.region.width + max,bpy.context.region.height/2))[0], V((bpy.context.region.width + max,bpy.context.region.height/2))[1])
	self.RU = Rot(self, V((bpy.context.region.width + max,bpy.context.region.height + max))[0], V((bpy.context.region.width + max,bpy.context.region.height + max))[1])
	self.C  = Rot(self, V((bpy.context.region.width/2,bpy.context.region.height /2 ))[0],  V((bpy.context.region.width/2,bpy.context.region.height /2 ))[1])
	self.CU = Rot(self, V((bpy.context.region.width/2,bpy.context.region.height + max ))[0], V((bpy.context.region.width/2,bpy.context.region.height + max ))[1])
	self.CD = Rot(self, V((bpy.context.region.width/2,zero))[0], V((bpy.context.region.width/2,zero))[1])

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
			if self.activeQuad[2]== "":
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
			if self.activeQuad[1]  == "white" and self.activeQuad[2]  == "red" and self.ModeRotation == True:
				self.ModeRotation = True
				self.addAngle = 180  # !!!!!!!!!!!
			elif self.activeQuad[1] == "white" and self.activeQuad[2] == "blue" and self.ModeRotation == True:
				self.ModeRotation = True
				self.addAngle = 180  # !!!!!!!!!!!
			elif self.activeQuad[1] == "green" and self.activeQuad[2] == "red" and self.ModeRotation == True:
				self.ModeRotation = True
				self.addAngle = 180  # !!!!!!!!!!!

			elif self.activeQuad[1]  == "white" and self.activeQuad[2] == "":
				self.ModeRotation = True
				self.addAngle = 180  # !!!!!!!!!!!



	elif blue:
		if self.activeQuad[0] != "blue":
			self.activeQuad[2] = self.activeQuad[1]
			self.activeQuad[1] = self.activeQuad[0]
			self.activeQuad[0] = "blue"
			if self.activeQuad[2]== "":
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
				self.addAngle = -180#!!!!!!!!!!!
			elif self.activeQuad[1] == "white" and  self.ModeRotation == True:
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

def CalculatePointForStartDrawing(self, context):
	self.width = 0
	for i in bpy.context.screen.areas:
		if i.type == 'VIEW_3D':
			for j in i.regions:
				if j.type == 'UI':
					width = j.width
	self.arr = []
	Offset = context.region.height * 0.05
	self.arr.append(V((0,0)))
	self.arr.append(V((bpy.context.region.width - width , 0)))
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
		glEnable(GL_BLEND)


		shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		batch = batch_for_shader(shader, 'TRIS', {"pos": self.arr})
		shader.bind()
		shader.uniform_float("color", (1, 1, 1, 0.1))
		batch.draw(shader)

		blf.position(0, ((context.region.width - self.width) / 2) - context.region.width * 0.1, (context.region.height * 0.025),0)
		blf.size(0, 30, 50)
		blf.draw(0, str(self.toolName))
		glDisable(GL_BLEND)
	except:
		pass

def DrawSCallBackZero(self, context):
	glEnable(GL_BLEND)
	shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
	batch = batch_for_shader(shader, 'TRIS', {"pos": self.mergeLine})
	shader.bind()
	shader.uniform_float("color", self.color)
	batch.draw(shader)

	shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
	batch = batch_for_shader(shader, 'TRIS', {"pos": self.arrowLeft})
	shader.bind()
	shader.uniform_float("color", self.color)
	batch.draw(shader)

	shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
	batch = batch_for_shader(shader, 'TRIS', {"pos": self.arrowRight})
	shader.bind()
	shader.uniform_float("color", self.color)
	batch.draw(shader)

	glLineWidth(2.5)
	shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
	batch = batch_for_shader(shader, 'LINES', {"pos": self.conturArrow})
	shader.bind()
	shader.uniform_float("color", (1.0, 1.0, 1.0, 0.4))
	batch.draw(shader)

	glLineWidth(2.5)
	shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
	batch = batch_for_shader(shader, 'LINES', {"pos": self.conturLine})
	shader.bind()
	shader.uniform_float("color", (1.0, 1.0, 1.0, 0.4))
	batch.draw(shader)

	if self.line != None:
		glLineWidth(4)
		shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		batch = batch_for_shader(shader, 'LINES', {"pos": self.line})
		shader.bind()
		shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
		batch.draw(shader)

		blf.position(0, self.mousePosition[0], self.mousePosition[1], 0)
		blf.size(0, 30, 50)
		blf.draw(0, str(self.axis))
	glDisable(GL_BLEND)

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

	merge = [v8, v9, v10,   v10, v11, v8]

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

	conturArrow = [v1,v2,v2,v3,v3,v6,v6,v7,v7,v4,v4,v5,v5,v1]
	conturLine = [v8,v9,v9, v10, v10, v11,v11,v8]





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
				self.mergeLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.ZeroMatrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_X_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_X_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))

		# elif self.axis == 'y':
			self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

			for i in merge:
				self.mergeLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.ZeroMatrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_Y_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_Y_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
		# elif self.axis == 'z':
			self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0, 0, 1)))).to_4x4()

			for i in merge:
				self.mergeLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.ZeroMatrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_Z_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_Z_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))

	if self.exc_axis == 'y':
		# if self.axis == 'x':
			self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()

			for i in merge:
				self.mergeLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.ZeroMatrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_X_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_X_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))

		# elif self.axis == 'y':
			self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 0.0)))).to_4x4()

			for i in merge:
				self.mergeLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.ZeroMatrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_Y_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_Y_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
		# elif self.axis == 'z':
			self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0, 0, 0)))).to_4x4()

			for i in merge:
				self.mergeLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.ZeroMatrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_Z_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_Z_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))

	if self.exc_axis == 'z':
		# if self.axis == 'x':
			self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()

			for i in merge:
				self.mergeLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.ZeroMatrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_X.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_X_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_X_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))

		# elif self.axis == 'y':
			self.ZeroMatrix = self.V_matrix# @ (Matrix.Rotation(1.570796, 3, Vector((0.0, 0.0, 1.0)))).to_4x4()

			for i in merge:
				self.mergeLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.V_matrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.V_matrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_Y.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.V_matrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_Y_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_Y_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
		# elif self.axis == 'z':
			self.ZeroMatrix = self.V_matrix @ (Matrix.Rotation(1.570796, 3, Vector((0, 0, 1)))).to_4x4()

			for i in merge:
				self.mergeLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, ((self.ZeroMatrix @ MS)) @ i))
			for i in conturLine:
				self.conturLine_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																		  ((self.ZeroMatrix @ MS)) @ i))
			for i in arrow:
				self.arrowLeft_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.arrowRight_Z.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))
			for i in conturArrow:
				self.conturArrow_Z_L.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
																			(((self.ZeroMatrix @ MS) @ MTL) @ MR) @ i))
				self.conturArrow_Z_R.append(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data,
															 ((self.ZeroMatrix @ MS) @ MTR) @ i))

def CalculateDrawMirror(self, context):
	for i in bpy.context.window.screen.areas:
		if i.type == 'VIEW_3D':
			CD = i.spaces[0].region_3d.view_distance / 20
			if CD >= 0.3:
				MS = Matrix.Scale(CD, 4, )
			else:
				MS = Matrix.Scale(0.3, 4, )

	v1 = V(( 0.0, -1.0 , 0.0))
	v2 = V(( 1.0,  0.0 , 0.0))
	v3 = V(( 0.4,  0.0 , 0.0))
	v4 = V((-0.4,  0.0 , 0.0))
	v5 = V((-1.0, -0.0 , 0.0))
	v6 = V(( 0.4,  3.0 , 0.0))
	v7 = V((-0.4,  3.0 , 0.0))

	arrow = [v1, v2, v3,
			 v1, v5, v4,
			 v4, v3, v1,
			 v3, v6, v7,
			 v4, v3, v7]
	conturArrow = [v1, v2, v2, v3, v3, v6, v6, v7, v7, v4, v4, v5, v5, v1]

	MR = Matrix.Rotation(1.570796 * 2, 3, Vector((0.0, 0.0, 1.0))).to_4x4()
	MT = Matrix.Translation(V((1.05, -1.7, 1.05)))
	a= []
	for i in arrow:
		a.append(MT @ i)
	arrow = a
	a = []
	for i in conturArrow:
		a.append(MT @ i)
	conturArrow = a
	a= []
	MT = Matrix.Translation(V((-1.05*2, -1.5*2, 0)))
	for i in arrow:
		a.append((MR @ MT) @ i)
	arrow += a
	a = []
	for i in conturArrow:
		a.append((MR @ MT) @ i)
	conturArrow += a
	a= []

	v8 =  V(( 1, 3.0 / 2, 1))
	v9 =  V((-1, 3.0 / 2, 1))
	v10 = V((-1, 3.0 / 2,-1))
	v11 = V(( 1, 3.0 / 2,-1))

	merge = [v8, v9, v10,   v10, v11, v8]
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
	glEnable(GL_BLEND)
	shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
	batch = batch_for_shader(shader, 'TRIS', {"pos": self.arrow})
	shader.bind()
	shader.uniform_float("color", self.color)
	batch.draw(shader)

	glLineWidth(2.5)
	shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
	batch = batch_for_shader(shader, 'LINES', {"pos": self.countr})
	shader.bind()
	shader.uniform_float("color", (1.0, 1.0, 1.0, 0.4))
	batch.draw(shader)

	if self.line != None:
		glLineWidth(4)
		shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
		batch = batch_for_shader(shader, 'LINES', {"pos": self.line})
		shader.bind()
		shader.uniform_float("color", (1.0, 1.0, 1.0, 0.3))
		batch.draw(shader)

		blf.position(0, self.mousePosition[0], self.mousePosition[1], 0)
		blf.size(0, 30, 50)
		blf.draw(0, str(self.axis))
	glDisable(GL_BLEND)






class AdvancedMove(Operator):
	''' Advanced move '''
	bl_idname = "view3d.advancedmove"
	bl_label = "Advanced Move"

	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"

	def modal(self, context, event):

		# -----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------------#

		if (event.type == 'LEFTMOUSE' or self.LB or self.SPACE) and (not event.shift and not event.alt):
			if self.SPACE:
				if event.value == 'RELEASE':
					UserPresets(self, context, True)
					UserSnap(self, context, Set=True)
					bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
					#context.area.header_text_set()
					return {'FINISHED'}
			else:
				if self.temp_loc_first is None:
					self.temp_loc_first = GetCoordMouse(self, context, event)
					self.LB = True
				if event.value == 'PRESS':
					if self.count_step <= 2:
						self.count_step += 1
						return {'RUNNING_MODAL'}
					else:
						self.temp_loc_last = GetCoordMouse(self, context, event)
						self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
						SetConstarin.SetMoveOnly(self, context, self.axis)
						self.LB = False
						self.temp_loc_first = None
						self.count_step = 0
						UserPresets(self, context, True)
						bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
						#context.area.header_text_set()
						return {'FINISHED'}



			# -----------------------RIGHT_MOUSE Exlude Axis-------------------------------------------------------------#

		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True
				SetConstarin.SetMoveExclude(self, context, self.exc_axis)
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
				# context.area.header_text_set()
				return {'FINISHED'}

			# -----------------------MIDDLE_MOUSE No Constrain-------------------------------------------------------------#

		elif event.type == 'MIDDLEMOUSE' or self.MB:
			self.MB = True
			if event.value == 'PRESS':
				SetConstarin.SetMoveNoConstrainNoSnap(self, context)
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
				# context.area.header_text_set()
				return {'FINISHED'}

			# -----------------------SPACEBAR Bottom            -------------------------------------------------------------#

		elif event.type == 'SPACE' or self.SPACE:
			self.SPACE = True

			if event.value == 'PRESS':
				SnapMoveOrientation(self, context)
				SetConstarin.SetMoveNoConstrain(self, context)
				return {'RUNNING_MODAL'}

		if event.shift or self.tweak:
			self.shift = True
			if event.type == 'LEFTMOUSE' or self.tweak:
				self.tweak = True
				if self.temp_loc_first is None:
					self.temp_loc_first = GetCoordMouse(self, context, event)
					if self.temp_loc_first == None:
						GetDirection(self)
						GetCenter(self)
						GetView(self)
						self.temp_loc_first = GetCoordMouse(self, context, event)
					if context.mode == "EDIT_MESH":
						bpy.ops.mesh.select_all(action='DESELECT')
					else:
						bpy.ops.object.select_all(action='DESELECT')
					bpy.ops.view3d.select('INVOKE_DEFAULT', extend=True, deselect=False, enumerate=False, toggle=False)
					#print ("sosok1")


				if self.count_step <= 2:
					self.count_step += 1
					#print(self.count_step)
					return {'RUNNING_MODAL'}
				else:
					self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					SetConstarin.SetMoveOnly(self, context, self.axis)
					self.tweak = False
					self.count_step = 0
					self.temp_loc_first = None
					return {'RUNNING_MODAL'}
		if (not event.shift) and self.shift:
			UserPresets(self, context, True)
			try:
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			except:
				pass
			context.area.tag_redraw()
			# context.area.header_text_set()
			return {'FINISHED'}

		if event.alt or self.tweakSelection:
			self.alt = True
			# print("Alt")
			if event.type == 'LEFTMOUSE' or self.tweakSelection:
				if self.temp_loc_first is None:
					self.temp_loc_first = GetCoordMouse(self, context, event)
					self.tweakSelection = True
					# print ("alt sosok1")
				if event.value == 'PRESS':
					# print("alt sosok2")
					if self.count_step <= 2:
						self.count_step += 1
						# print(self.count_step)
						return {'RUNNING_MODAL'}
					else:
						# print("alt sosok4")
						self.temp_loc_last = GetCoordMouse(self, context, event)
						self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
						SetConstarin.SetMoveOnly(self, context, self.axis)
						self.tweakSelection = False
						self.count_step = 0
						self.temp_loc_first = None
					return {'RUNNING_MODAL'}
		if (not event.alt) and self.alt:
			# print ("alt  exit")
			UserPresets(self, context, True)
			try:
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			except:
				pass
			context.area.tag_redraw()
			# context.area.header_text_set()
			return {'FINISHED'}

		elif event.unicode == 'G' or event.unicode == 'g':
			if context.mode == "EDIT_MESH":
				bpy.ops.transform.edge_slide('INVOKE_DEFAULT')
				UserPresets(self, context, True)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
				# context.area.header_text_set()
				return {'FINISHED'}
		elif event.unicode == 'X' or event.unicode == 'x':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			# context.area.header_text_set()
			return {'FINISHED'}
		elif event.unicode == 'Y' or event.unicode == 'y':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			# context.area.header_text_set()
			return {'FINISHED'}
		elif event.unicode == 'Z' or event.unicode == 'z':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			# context.area.header_text_set()
			return {'FINISHED'}

		if event.type == 'ESC':
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			# context.area.header_text_set()
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		context.area.header_text_set(
			'Drag LMB constraint axis, RMB translate by two axis, MMB free translate, SPACE free translate with snap and rotate along normal, Shift tweak new selection, Alt Current selection')
		if context.space_data.type == 'VIEW_3D':
			UserSnap(self, context)
			UserPresets(self, context)
			GetDirection(self)
			GetCenter(self)
			GetView(self)
			CalculatePointForStartDrawing(self, context)

			self.LB = False
			self.RB = False
			self.MB = False
			self.SPACE = False
			self.tweak = False
			self.temp_loc_first = None
			self.temp_loc_last = None
			self.count_step = 0
			self.toolName = "Advanced Move"
			self.W = False
			self.shift = False
			self.exit = False
			self.alt = False
			self.tweakSelection = False

			argsStart = (self, context)
			self._handle1 = bpy.types.SpaceView3D.draw_handler_add(DrawStartTool, argsStart, 'WINDOW', 'POST_PIXEL')

			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
			return {'CANCELLED'}

class AdvancedScale(Operator):
	''' Advanced Scale '''
	bl_idname = "view3d.advancedscale"
	bl_label = "Advanced Scale"
	#bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"

	def modal(self, context, event):
		# -----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------------#

		if event.type == 'LEFTMOUSE' or self.LB:
			if self.temp_loc_first is None:
				self.temp_loc_first = GetCoordMouse(self, context, event)
			self.LB = True
			if event.value == 'PRESS':
				if self.count_step <= 2:
					self.count_step += 1
					return {'RUNNING_MODAL'}
				else:
					self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					SetConstarin.SetScaleOnly(self, context, self.axis)
				return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
				return {'FINISHED'}

			# -----------------------RIGHT_MOUSE Exlude Axis-------------------------------------------------------------#

		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True
				SetConstarin.SetScaleExclude(self, context, self.exc_axis)
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
				return {'FINISHED'}

			# -----------------------MIDDLE_MOUSE No Constrain-------------------------------------------------------------#

		elif event.type == 'MIDDLEMOUSE' or self.MB:
			self.MB = True
			if event.value == 'PRESS':
				SetConstarin.SetScaleNoConstrain(self, context)
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
				return {'FINISHED'}

			# -----------------------SPACEBAR Bottom Setup zero-------------------------------------------------------------#

		elif event.type == 'SPACE' or self.SPACE:
			bpy.ops.view3d.advancedscale_zero('INVOKE_DEFAULT')
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			return {'FINISHED'}

				# -----------------------ALT for negative value-------------------------------------------------------------#

		# elif event.type == 'BUTTON4MOUSE' or self.ALT:
		elif event.shift or self.ALT:
			bpy.ops.view3d.advancedscale_mirror('INVOKE_DEFAULT')
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			return {'FINISHED'}

		elif event.unicode == 'X' or event.unicode == 'x':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, False))
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			return {'FINISHED'}
		elif event.unicode == 'Y' or event.unicode == 'y':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, False))
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			return {'FINISHED'}
		elif event.unicode == 'Z' or event.unicode == 'z':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, False, True))
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			return {'FINISHED'}

		if event.type == 'ESC':
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		context.area.header_text_set('Drag LMB constraint axis, RMB resize by two axis, MMB free resize, SHIFT mirror, SPACE flatten')
		if context.space_data.type == 'VIEW_3D':
			UserSnap(self, context)
			UserPresets(self, context)
			GetDirection(self)
			GetCenter(self)
			GetView(self)
			CalculatePointForStartDrawing(self, context)
			self.LB = False
			self.RB = False
			self.MB = False
			self.SPACE = False
			self.ALT = False
			self.temp_loc_first = None
			self.temp_loc_last = None
			self.count_step = 0
			self.toolName = "Advanced Scale"

			argsStart = (self, context)
			self._handle1 = bpy.types.SpaceView3D.draw_handler_add(DrawStartTool, argsStart, 'WINDOW', 'POST_PIXEL')

			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
			return {'CANCELLED'}

class AdvancedScaleZore(Operator):
	''' Advanced Scale '''
	bl_idname = "view3d.advancedscale_zero"
	bl_label = "Advanced Scale zero"
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"

	def modal(self, context, event):
		self.temp_loc_last = GetCoordMouse(self, context, event)
		self.mousePosition = event.mouse_region_x, event.mouse_region_y
		self.line = [(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)),
				(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.temp_loc_last))]
		self.axis = SetupAxis(self, self.center, self.temp_loc_last)
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
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle_Zero, 'WINDOW')
			context.area.tag_redraw()
			return {'FINISHED'}


		if event.type == 'ESC':
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle_Zero, 'WINDOW')
			context.area.tag_redraw()
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
	   # context.area.header_text_set('Drag LMB constraint axis, RMB resize by two axis, MMB free resize, SHIFT mirror, SPACE flatten')
		if context.space_data.type == 'VIEW_3D':
			UserSnap(self, context)
			UserPresets(self, context)
			GetDirection(self)
			GetCenter(self)
			GetView(self)

			self.temp_loc_last = GetCoordMouse(self, context, event)
			self.mousePosition = event.mouse_region_x, event.mouse_region_y
			self.line = [
				(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)),
				(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.temp_loc_last))
				]
			self.axis = SetupAxis(self, self.center, self.temp_loc_last)
			CalculateDrawZero(self, context)
			self.line = None
			self.color = V((0.8, 0.4 ,0.0 , 0.3))

			argsStart = (self, context)
			self._handle_Zero = bpy.types.SpaceView3D.draw_handler_add(DrawSCallBackZero, argsStart, 'WINDOW', 'POST_PIXEL')

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
		self.temp_loc_last = GetCoordMouse(self, context, event)
		self.mousePosition = event.mouse_region_x, event.mouse_region_y
		self.line = [(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)),
				(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.temp_loc_last))]
		self.axis = SetupAxis(self, self.center, self.temp_loc_last)
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
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle_Mirror, 'WINDOW')
			context.area.tag_redraw()
			return {'FINISHED'}


		if event.type == 'ESC':
			UserPresets(self, context, True)
			bpy.types.SpaceView3D.draw_handler_remove(self._handle_Mirror, 'WINDOW')
			context.area.tag_redraw()
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
	   # context.area.header_text_set('Drag LMB constraint axis, RMB resize by two axis, MMB free resize, SHIFT mirror, SPACE flatten')
		if context.space_data.type == 'VIEW_3D':
			UserSnap(self, context)
			UserPresets(self, context)
			GetDirection(self)
			GetCenter(self)
			GetView(self)

			self.temp_loc_last = GetCoordMouse(self, context, event)
			self.mousePosition = event.mouse_region_x, event.mouse_region_y
			self.line = [
				(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.center)),
				(view3d_utils.location_3d_to_region_2d(bpy.context.region, bpy.context.region_data, self.temp_loc_last))
				]
			self.axis = SetupAxis(self, self.center, self.temp_loc_last)
			CalculateDrawMirror(self, context)
			self.line = None
			self.color = V((0.8, 0.4 ,0.0 , 0.3))

			argsStart = (self, context)
			self._handle_Mirror = bpy.types.SpaceView3D.draw_handler_add(DrawScaleMirror, argsStart, 'WINDOW', 'POST_PIXEL')

			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
			return {'CANCELLED'}

class AdvancedRotation(Operator):
	''' Advanced move '''
	bl_idname = "view3d.advanced_rotation"
	bl_label = "Advanced Rotation"
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"

	def modal(self, context, event):
		context.area.tag_redraw()
		self.mousePosition = event.mouse_region_x, event.mouse_region_y
		self.temp_loc_last = GetCoordMouse(self, context, event)

		# bpy.ops.wm.redraw_timer(type='DRAW', iterations=1)

		if event.type == 'LEFTMOUSE' or self.LB:
			if not context.preferences.addons[__name__].preferences.Use_Advanced_Transform:
				SetConstarin.SetRotationOnly(self, context, self.exc_axis)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
				UserPresets(self, context, True)
				context.area.tag_redraw()
				return {'FINISHED'}
			else:
				if self.temp_loc_first is None:
					self.temp_loc_first = GetCoordMouse(self, context, event)
					StartDraw(self, context, event)
					CalculateCirclePoint(self)
					args = (self, context, event)
					self._handle = bpy.types.SpaceView3D.draw_handler_add(DrawCalback, args, 'WINDOW', 'POST_PIXEL')
					self.LB = True
					self.Step = 1


				if event.value == 'PRESS':
					intersect_Quad(self, event)
					if CheckRotation(self):
						rotation = self.Angle - self.LastRotation
						self.LastRotation = self.Angle

						if self.negativMlt:
							SetConstarin.SnapRotation(self, context, rotation*-1, self.exc_axis)
						else:
							SetConstarin.SnapRotation(self, context, rotation, self.exc_axis)
					return {'RUNNING_MODAL'}
				elif event.value == 'RELEASE':
					bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
					bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
					UserPresets(self, context, True)
					context.area.tag_redraw()
					return {'FINISHED'}

		elif event.type == 'RIGHTMOUSE' or self.RB:
			#self.temp_loc_last = GetCoordMouse(self, context, event)
			if self.temp_loc_first is None:
				self.temp_loc_first = GetCoordMouse(self, context, event)
				StartDraw(self, context, event)
				CalculateCirclePoint(self)
				args = (self, context, event)
				self._handle = bpy.types.SpaceView3D.draw_handler_add(DrawCalback, args, 'WINDOW', 'POST_PIXEL')
				self.RB = True
			if event.value == 'PRESS':
				intersect_Quad(self, event)
				if CheckRotation(self):
					rotation = self.Angle - self.LastRotation
					self.LastRotation = self.Angle

					if self.negativMlt:
						SetConstarin.SnapRotation(self, context, rotation*-1, self.exc_axis)
					else:
						SetConstarin.SnapRotation(self, context, rotation, self.exc_axis)
				return {'RUNNING_MODAL'}
			elif event.value == 'RELEASE':
				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
				bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
				UserPresets(self, context, True)
				context.area.tag_redraw()
				return {'FINISHED'}

		if event.type == 'ESC':
			UserPresets(self, context, True)
			try:
				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
			except:
				pass
			bpy.types.SpaceView3D.draw_handler_remove(self._handle1, 'WINDOW')
			context.area.tag_redraw()
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		context.area.header_text_set('LMB constraint view axis, RMB constraint view axis snap, MMB free rotate')
		# self.time = time.clock()
		if context.space_data.type == 'VIEW_3D':
			PREFS = context.preferences.addons[__name__].preferences
			self.LB = False
			self.LB_cal = True
			self.RB = False
			self.MB = False
			self.SPACE = False
			self.temp_loc_first = None
			self.temp_loc_last = None
			self.mousePosition = None
			self.count_step = 0
			self.Angle = 0
			self.directionMatrix = None
			self.LastRotation = 0
			self.Step = int(context.preferences.addons[__name__].preferences.Snapping_Step)
			self.FullCircle = False
			self.activeQuad = ["","",""]
			self.ModeRotation = True # Если True использовать позитивные значение
			self.stepDivision = 360 / self.Step
			self.addAngle = 0
			self.NegativeMatrix = Matrix()
			self.negativMlt = False
			self.realAngle = 0
			self.is_snap = True
			self.call_no_snap = 0
			self.toolName = "Advanced Rotation"



			UserSnap(self, context)
			UserPresets(self, context)
			GetDirection(self)
			GetCenter(self)
			GetView(self)
			CalculatePointForStartDrawing(self, context)

			argsStart = (self, context)
			self._handle1 = bpy.types.SpaceView3D.draw_handler_add(DrawStartTool, argsStart, 'WINDOW', 'POST_PIXEL')




			self.steps = [0]
			for i in range(0, 72):
				self.steps.append(self.steps[-1] + 5)
			# print("self.steps",self.steps[:])


			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
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
		# -----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------------#

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
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				return {'FINISHED'}

			# -----------------------RIGHT_MOUSE Exlude Axis-------------------------------------------------------------#

		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True
				SetConstarin.SetMoveExcludeUV(self)
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				return {'FINISHED'}
		#
		#     # -----------------------MIDDLE_MOUSE No Constrain-------------------------------------------------------------#
		#
		# elif event.type == 'MIDDLEMOUSE' or self.MB:
		#     self.MB = True
		#     if event.value == 'PRESS':
		#         SetConstarin.SetMoveNoConstrainNoSnap(self, context)
		#     if event.value == 'RELEASE':
		#         UserPresets(self, context, False)
		#         return {'FINISHED'}
		#
		#     # -----------------------SPACEBAR Bottom            -------------------------------------------------------------#
		#
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
		#         UserPresets(self, context, True)
		#         return {'FINISHED'}
		# elif event.unicode == 'X' or event.unicode == 'x':
		#     bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		#     UserPresets(self, context, True)
		#     return {'FINISHED'}
		# elif event.unicode == 'Y' or event.unicode == 'y':
		#     bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
		#     UserPresets(self, context, True)
		#     return {'FINISHED'}
		# elif event.unicode == 'Z' or event.unicode == 'z':
		#     bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
		#     UserPresets(self, context, True)
		#     return {'FINISHED'}

		if event.type == 'ESC':
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		#context.area.header_text_set('Drag LMB constraint axis, RMB translate by two axis, MMB free translate, SPACE free translate with snap and rotate along normal')
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

			#args = (self, context)
			#self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_VIEW')
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
		# -----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------------#

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
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				return {'FINISHED'}

			# -----------------------RIGHT_MOUSE Exlude Axis-------------------------------------------------------------#

		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True
				SetConstarin.SetScaleExcludeUV(self, context, self.exc_axis)
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				return {'FINISHED'}


			# -----------------------SPACEBAR Bottom Setup zero-------------------------------------------------------------#

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
					UserPresets(self, context, True)
					return {'FINISHED'}

				# -----------------------ALT for negative value-------------------------------------------------------------#

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
					UserPresets(self, context, True)
					return {'FINISHED'}

		elif event.unicode == 'X' or event.unicode == 'x':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, False))
			UserPresets(self, context, True)
			return {'FINISHED'}
		elif event.unicode == 'Y' or event.unicode == 'y':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, False))
			UserPresets(self, context, True)
			return {'FINISHED'}

		if event.type == 'ESC':
			UserPresets(self, context, True)
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
	   # context.area.header_text_set('Drag LMB constraint axis, RMB resize by two axis, MMB free resize, SHIFT mirror, SPACE flatten')
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

			#args = (self, context)
			#self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_VIEW')
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
			if event.value == 'RELEASE':
				UserPresets(self, context, True)
				return {'FINISHED'}

		elif event.type == 'RIGHTMOUSE' or self.RB:
			self.temp_loc_last = GetCoordMouse(self, context, event)
			if self.temp_loc_first is None:
				self.temp_loc_first = GetCoordMouse(self, context, event)

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
			if event.value == 'RELEASE':
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
			UserPresets(self, context, True)
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		#context.area.header_text_set('LMB constraint view axis, RMB constraint view axis snap 15 degrees, MMB free rotate')
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
			#args = (self, context)
			#self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_rot, args, 'WINDOW', 'POST_PIXEL')
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
	'''
	returns hotkey of specific type, with specific properties.name (keymap is not a dict, so referencing by keys is not enough
	if there are multiple hotkeys!)
	'''
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
	#kmi.properties.name = "view3d.advancedmove"
	kmi.active = True
	addon_keymaps.append((km, kmi))

	wm1 = bpy.context.window_manager
	kc1 = wm1.keyconfigs.addon
	km1 = kc1.keymaps.new(name="3D View Generic", space_type='VIEW_3D', region_type='WINDOW')
	kmi1 = km1.keymap_items.new(AdvancedScale.bl_idname, 'S', 'PRESS', shift=False, ctrl=False, alt=False)
	#kmi.properties.name = "view3d.advancedmove"
	kmi1.active = True
	addon_keymaps.append((km1, kmi1))

	wm2 = bpy.context.window_manager
	kc2 = wm2.keyconfigs.addon
	km2 = kc2.keymaps.new(name="3D View Generic", space_type='VIEW_3D', region_type='WINDOW')
	kmi2 = km2.keymap_items.new(AdvancedRotation.bl_idname, 'R', 'PRESS', shift=False, ctrl=False, alt=False)
	#kmi.properties.name = "view3d.advancedmove"
	kmi2.active = True
	addon_keymaps.append((km2, kmi2))

class AdvancedTransform_Add_Hotkey(bpy.types.Operator):
	''' Add hotkey entry '''
	bl_idname = "advanced_transform.add_hotkey"
	bl_label = "Advanced Transform Add Hotkey"
	bl_options = {'REGISTER', 'INTERNAL'}

	def execute(self, context):
		add_hotkey()

		self.report({'INFO'}, "Hotkey added in User Preferences -> Input -> Screen -> Screen (Global)")
		return {'FINISHED'}


def remove_hotkey():
	''' clears all addon level keymap hotkeys stored in addon_keymaps '''
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
	Snapping_Step = EnumProperty(
		items=[('5', "5", "Snapping Step" ),
			   ('10', "10", "Snapping Step"),
			   ('15', "15", "Snapping Step"),
			   ('30', "30", "Snapping Step"),
			   ('45', "45", "Snapping Step"),
			   ('90', "90", "Snapping Step")
			],
		name="Rotation Snapping Step",
		default='15',
		# update=use_cashes
	#caches_valid = True
	)
	Use_Advanced_Transform = BoolProperty(
			name="Use Advanced Transform Rotation",
			default=False,
			description = "Is not use standard blender rotation without snapping (left mouse button).Standard has great performance.",
			)
	def draw(self, context):
		layout = self.layout
		layout.prop(self, "Snapping_Step")
		layout.prop(self, "Use_Advanced_Transform")
		#---------------------------------
		box = layout.box()
		split = box.split()
		col = split.column()
		#col.label("Setup Advanced Move")
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
		#-----------------------------------
		box1 = layout.box()
		split1 = box1.split()
		col1 = split1.column()
		# col.label("Setup Advanced Move")
		col1.separator()
		wm1 = bpy.context.window_manager
		kc1 = wm1.keyconfigs.user
		km1 = kc1.keymaps['3D View Generic']
		kmi1 = get_hotkey_entry_item(km1, "view3d.advancedscale" )
		if kmi1:
			col1.context_pointer_set("keymap", km1)
			rna_keymap_ui.draw_kmi([], kc1, km1, kmi1, col1, 0)
		else:
			col1.label("No hotkey entry found")
			col1.operator(AdvancedTransform_Add_Hotkey.bl_idname, text="Add hotkey entry", icon='ZOOMIN')
		#--------------------------------------
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


class SetConstarin(Operator):
	bl_idname = "view3d.setconstrain"
	bl_label = "SetConstrain"

	# -----------Constrain for move-----------#

	def SetMoveOnly(self, context, axsis):
		result = None
		if self.axis == 'x':
			result = bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		elif self.axis == 'y':
			result = bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
		elif self.axis == 'z':
			result = bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
		# print(result)
		# return result

	def SetMoveExclude(self, context, axis):
		if self.exc_axis == 'x':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, True))
		elif self.exc_axis == 'y':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, True))
		elif self.exc_axis == 'z':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, True, False))
		return {'FINISHED'}

	def SetMoveNoConstrain(self, context):
		bpy.ops.transform.translate('INVOKE_DEFAULT', snap=True, snap_target='CENTER', snap_align=True)
		return {'FINISHED'}

	def SetMoveNoConstrainNoSnap(self, context):

		bpy.ops.transform.translate('INVOKE_DEFAULT')
		return {'FINISHED'}

	# --------Constrain for rotation----------#

	def SetRotationOnlyAT(self, context, angle, axis):
		if axis == 'x':
			bpy.ops.transform.rotate(value=math.radians(angle), orient_axis='X', constraint_axis=(True, False, False))
		elif axis == 'y':
			bpy.ops.transform.rotate(value=math.radians(angle), orient_axis='Y', constraint_axis=(False, True, False))
		elif axis == 'z':
			bpy.ops.transform.rotate(value=math.radians(angle), orient_axis='Z', constraint_axis=(False, False, True))
		return {'FINISHED'}

	def SetRotationOnly(self, context, axis):
		if self.exc_axis == 'x':
			bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		elif self.exc_axis == 'y':
			bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
		elif self.exc_axis == 'z':
			bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
		return {'FINISHED'}

	# -----------Constrain for scale-----------#

	def SetScaleOnly(self, context, axis):
		if self.axis == 'x':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		elif self.axis == 'y':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, False, True))
		return {'FINISHED'}

	def SetScaleExclude(self, context, axis):
		if self.exc_axis == 'x':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, True))
		elif self.exc_axis == 'y':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, True))
		elif self.exc_axis == 'z':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, True, False))
		return {'FINISHED'}

	def SetScaleNoConstrain(self, context):
		bpy.ops.transform.resize('INVOKE_DEFAULT')
		return {'FINISHED'}

	# -----------Advanced Scale Constrain-----------#

	# -----------Align to 0-----------#

	def SetScaleOnlySetZero(self, context, axis):
		if self.axis == 'x':
			bpy.ops.transform.resize(value=(0.0, 1.0, 1.0), constraint_axis=(True, False, False))
		elif self.axis == 'y':
			bpy.ops.transform.resize(value=(1.0, 0.0, 1.0), constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.resize(value=(1.0, 1.0, 0.0), constraint_axis=(False, False, True))
		return {'FINISHED'}

	def SetScaleExcludeSetZero(self, context, axis):
		if self.axis == 'x':
			bpy.ops.transform.resize(value=(0.0, 1.0, 1.0), constraint_axis=(False, True, True))
		elif self.axis == 'y':
			bpy.ops.transform.resize(value=(1.0, 0.0, 1.0), constraint_axis=(True, False, True))
		elif self.axis == 'z':
			bpy.ops.transform.resize(value=(1.0, 1.0, 0.0), constraint_axis=(True, True, False))
		return {'FINISHED'}

	# -----------Align to -1 -----------#

	def SetScaleOnlySetNegative(self, context, axis):
		if self.axis == 'x':
			bpy.ops.transform.resize(value=(-1.0, 1.0, 1.0), constraint_axis=(True, False, False))
		elif self.axis == 'y':
			bpy.ops.transform.resize(value=(1.0, -1.0, 1.0), constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.resize(value=(1.0, 1.0, -1.0), constraint_axis=(False, False, True))
		return {'FINISHED'}

	# -----------Advanced rotation Constrain-----------#
	# -----------Step rotation 45----------#52jh

	def SetRotationOnlyStepgNegative(self, context, axis):
		if self.axis == 'x':
			bpy.ops.transform.rotate(value=0.785398, constraint_axis=(True, False, False))
		elif self.axis == 'y':
			bpy.ops.transform.rotate(value=0.785398, constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.rotate(value=0.785398, constraint_axis=(False, False, True))
		return {'FINISHED'}

	def SetRotationOnlyStepgPositive(self, context, axis):
		if self.axis == 'x':
			bpy.ops.transform.rotate(value=-0.785398, constraint_axis=(True, False, False))
		elif self.axis == 'y':
			bpy.ops.transform.rotate(value=-0.785398, constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.rotate(value=-0.785398, constraint_axis=(False, False, True))
		return {'FINISHED'}

	def SnapRotation(self, context, angle, axis):
		if axis == 'x':
			bpy.ops.transform.rotate(value=math.radians(angle), orient_axis='X', constraint_axis=(True, False, False))
		elif axis == 'y':
			bpy.ops.transform.rotate(value=math.radians(angle), orient_axis='Y', constraint_axis=(False, True, False))
		elif axis == 'z':
			bpy.ops.transform.rotate(value=math.radians(angle), orient_axis='Z', constraint_axis=(False, False, True))
		return {'FINISHED'}
	#--------------------UV---------------------#
	def SetMoveOnlyUV(self, axis):
		if axis == 'x':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		elif axis == 'y':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
			return {'FINISHED'}
	def SetMoveExcludeUV(self):
		bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, True, False))

classes = (AdvancedMove, AdvancedScale, AdvancedRotation, AdvancedMoveUV, AdvancedScaleUV, AdvancedRotationUV, AdvancedTransformPref, AdvancedScaleZore, AdvancedScaleMirror, AdvancedTransform_Add_Hotkey)

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
