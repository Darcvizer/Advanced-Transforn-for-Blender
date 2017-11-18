bl_info = {
	"name": "Advanced Transform",
	"location": "View3D > Add > Object > Transform,",
	"description": "Auto set axis constrain",
	"author": "Vladislav Kindushov",
	"version": (0, 2),
	"blender": (2, 7, 8),
	"category": "Object",
}

import bpy
import blf
import bgl
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix
from bpy.props import IntProperty
from bpy.types import Operator, Macro
from bpy.props import IntProperty, FloatProperty
from mathutils.geometry import intersect_line_plane
import numpy as np
import math
import time

#-------		For fix looping -----------#

def UserPresets(self, context, chen):
	global save_user_drag
	#bpy.app.binary_path

	if chen == True:
		save_user_drag = context.user_preferences.edit.use_drag_immediately
		if save_user_drag != True:
			context.user_preferences.edit.use_drag_immediately = True

	elif chen == False:
		context.user_preferences.edit.use_drag_immediately = save_user_drag

# --------  		Create transform orientation			-------#

def CreateOrientation(self, context):
	'''Create new Transform Orientation'''
	temp_name = "AdvancedTansform"
	temp = bpy.context.scene.orientations.find(temp_name)
	if temp_name != 'GLOBAL':
		if temp == -1:
			bpy.ops.transform.create_orientation(name = temp_name, use = True, overwrite = True)
		#bpy.context.space_data.transform_orientation = temp_name

		user_matrix = context.scene.orientations[temp_name].matrix

	return user_matrix

#--------   		Delete self addon orientation   		-------#

def DeleteOrientation(self, context,):
	'''Delete self addon orientation'''
	temp_name = "AdvancedTansform"
	try:
		#if user_orient != 'GLOBAL':
		context.space_data.transform_orientation = temp_name
		bpy.ops.transform.delete_orientation('INVOKE_DEFAULT')
		#else:
		return {'FINISHED'}
	except:
		return {'FINISHED'}

#---------  			Get user snap setting   			-------#

def GetUserSnap(self, context):
	global snap_user
	global user_snap_element
	global user_snap_target
	global user_snap_rot
	global user_snap_project
	global user_snap

	user_snap_element = context.scene.tool_settings.snap_element
	user_snap_target =  context.scene.tool_settings.snap_target
	user_snap_rot = 	context.scene.tool_settings.use_snap_align_rotation
	user_snap_project = context.scene.tool_settings.use_snap_project
	user_snap = 		context.scene.tool_settings.use_snap

#---------  			Set user snap setting   			-------#

def SetUserSnap(self, context):
	context.scene.tool_settings.snap_element =  			user_snap_element
	context.scene.tool_settings.snap_target =   			user_snap_target
	context.scene.tool_settings.use_snap_align_rotation =   user_snap_rot
	context.scene.tool_settings.use_snap_project =  		user_snap_project
	context.scene.tool_settings.use_snap =  				user_snap

# --------- 			Set snap setting  for move  		-------#

def SnapMoveOrientation(self, context):
	context.scene.tool_settings.snap_element =  			'FACE'
	context.scene.tool_settings.snap_target =   			'CENTER'
	context.scene.tool_settings.use_snap_align_rotation =   True
	context.scene.tool_settings.use_snap_project =  		True
	context.scene.tool_settings.use_snap =  				True

# --------  		Save user transform orientation 		-------#

def SaveOrientation(self, context):
	global user_orient
	user_orient = bpy.context.space_data.transform_orientation
	print("orinent",user_orient)
	return user_orient



def getView(context, event):
	"""Get Viewport Vector""" 
	region = context.region
	rv3d = context.region_data
	return rv3d.view_rotation * Vector((0.0, 0.0, -1.0))

def CreateMatrixByView(self, context, event):
	vector = getView(context, event)
	if self.matrix:
		vector = self.matrix.to_3x3().inverted() * vector
	else:
		self.matrix = Matrix()
	x = vector[0]
	if x < 0:
		x = -x
	y = vector[1]
	if y < 0:
		y = -y
	z = vector[2]
	if z < 0:
		z = -z

	if x > y and x > z:
		self.g_matrix = Matrix.Translation(self.center) * (Matrix(self.matrix.to_3x3() * Matrix.Rotation(1.570796,3,Vector((0.0,1.0,0.0)))).to_4x4())
		self.exc_axis = 'x'
	elif y > x and y > z:
		self.g_matrix = Matrix.Translation(self.center) * (Matrix(self.matrix.to_3x3() * Matrix.Rotation(1.570796,3,Vector((1.0,0.0,0.0)))).to_4x4())
		self.exc_axis = 'y'
	elif z > x and z > y:
		self.g_matrix = Matrix.Translation(self.center) * (Matrix(self.matrix.to_3x3() * Matrix.Rotation(1.570796,3,Vector((0.0,0.0,1.0)))).to_4x4())
		self.exc_axis = 'z'
	

def draw_callback_px(self, context):
	try:
		mw = self.g_matrix
	

		scale = 1.0
		bgl.glEnable(bgl.GL_BLEND)
		bgl.glColor4f(0.471938, 0.530946, 0.8, 0.06)
		bgl.glBegin(bgl.GL_POLYGON)
	
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale * Vector((1.0,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale * Vector((-1.0,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale * Vector((-1.0,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale * Vector((1.0,-1.0,0.0))))))
		bgl.glEnd()
	
	
		#bgl.glEnable(bgl.GL_BLEND)
		bgl.glLineWidth(1.0)
		bgl.glBegin(bgl.GL_LINES)
		bgl.glColor4f(1, 1, 1, 0.1)
	
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.8,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.8,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.6,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.6,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.4,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.4,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.2,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.2,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.0,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((0.0,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-0.2,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-0.2,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-0.4,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-0.4,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-0.6,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-0.6,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-0.8,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-0.8,-1.0,0.0))))))
		bgl.glColor4f(1, 1, 1, 0.1)
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,0.8,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,0.8,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,0.6,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,0.6,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,0.4,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,0.4,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,0.2,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,0.2,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,0.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,0.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,-0.2,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,-0.2,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,-0.4,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,-0.4,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,-0.6,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,-0.6,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,-0.8,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,-0.8,0.0))))))
		bgl.glEnd()
		bgl.glLineWidth(0.6)
		bgl.glBegin(bgl.GL_LINES)
	
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,-1.0,0.0))))))
	
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,-1.0,0.0))))))
	
	
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,1.0,0.0))))))
	
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((1.0,-1.0,0.0))))))
		bgl.glVertex3f(*(mw *((Zoom(self, context)/2)*(scale *Vector((-1.0,-1.0,0.0))))))
	
		bgl.glEnd()
	
		# bgl.glBegin(bgl.GL_POLYGON)
		# bgl.glColor3f(1.0, 0.0, 0.0)
		# steps= 40
		# for step in range(steps):
		#     a = (math.pi * 2 / steps) * step
		#     glVertex3f(*(mw * Vector((0 + radius * math.cos(a), 0 + radius * math.sin(a), 0.0))))
		# glEnd()
	
	
		bgl.glEnd()
	
		# restore opengl defaults
		bgl.glLineWidth(1)
		bgl.glDisable(bgl.GL_BLEND)
		bgl.glColor4f(0.0, 0.0, 0.0, 1.0)
	except:
		pass
	
def draw_callback_rot(self, context):
	##try:
	mw = self.g_matrix
	vec = None
	vec2 = None
	point1 = None
	point2 = None
	point3 = None
	bgl.glEnable(bgl.GL_BLEND)
	if not self.temp_loc_first is None:
		point1 = self.center.copy()
		vec = (self.temp_loc_first.copy() - self.center.copy()).normalized()
		point2 = point1.copy()
		point2 += vec * 3
		point3 = point1.copy()
		vec2 = (self.temp_loc_last.copy() - self.center.copy()).normalized()
		point3 += vec2 * 3
		
		axis_dst = Vector((0.0, 1.0, 0.0))
		matrix_rotate = Matrix.Rotation(1.570796,3,Vector((0.0,0.0,1.0))).to_4x4() * (axis_dst.rotation_difference(self.g_matrix.to_3x3().inverted() * (vec * 1.570796)).to_matrix().to_4x4())
		mw = mw * matrix_rotate

		bgl.glEnable(bgl.GL_BLEND)
		bgl.glColor4f(1, 1, 1, 0.5)
		
		bgl.glLineWidth(1)
		bgl.glBegin(bgl.GL_LINES)
		
		
		bgl.glVertex3f(*point1)
		bgl.glVertex3f(*point2)
		
		bgl.glVertex3f(*point1)
		bgl.glVertex3f(*point3)
		bgl.glEnd()
		
		angl = math.degrees(vec.angle(vec2))
		
		bgl.glBegin(bgl.GL_POLYGON)
		bgl.glColor4f(0.471938, 0.530946, 0.8, 0.2)
		radius = 3
		steps = 360
		bgl.glVertex3f(*self.center)
		
		v = (mw * (Vector((0 + radius * math.cos((math.pi * 2 / steps) * (round(angl)+1)), 0 + radius * math.sin((math.pi * 2 / steps) * (round(angl)+1)), 0.0))))
		
		if (point3 - v).length > 0.01:
			i = round(angl) + 1
			b = 360
			while b != 360-i:
				a = (math.pi * 2 / steps) * b
				bgl.glVertex3f(*(mw * (Vector((0 + radius * math.cos(a), 0 + radius * math.sin(a), 0.0)))))
				b -= 1
			bgl.glEnd()
			
				
		else:
			for step in range((round(angl)+1)):
				a = (math.pi * 2 / steps) * step
				bgl.glVertex3f(*(mw * (Vector((0 + radius * math.cos(a), 0 + radius * math.sin(a), 0.0)))))
			bgl.glEnd()
	
	else:
		bgl.glBegin(bgl.GL_POLYGON)
		bgl.glColor4f(0.471938, 0.530946, 0.8, 0.2)
		radius = 3
		steps= 90
		bgl.glVertex3f(*self.center)
		for step in range(46):
			a = (math.pi * 2 / steps) * step
		bgl. glVertex3f(* (mw *(Vector((0 + radius * math.cos(a), 0 + radius * math.sin(a), 0.0)))))
		bgl.glEnd()
	#
	# 	# restore opengl defaults
	bgl.glLineWidth(1)
	bgl.glDisable(bgl.GL_BLEND)
	bgl.glColor4f(0.0, 0.0, 0.0, 1.0)
	#except:
	#pass

def GetCoordMouse(self, context, event, point=None, matrix=None, revers=False):
	""" 
	convert mouse pos to 3d point over plane defined by origin and normal 
	"""
	point = self.center
	matrix = self.g_matrix
	# get the context arguments
	region = bpy.context.region
	rv3d = bpy.context.region_data
	coord = event.mouse_region_x, event.mouse_region_y
	view_vector_mouse = view3d_utils.region_2d_to_vector_3d(region, rv3d,coord)
	ray_origin_mouse = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
	loc = intersect_line_plane(ray_origin_mouse, ray_origin_mouse + view_vector_mouse, point, matrix.to_3x3() * Vector((0.0,0.0,1.0)), False)
	return loc

# --------  		Get axis for only transform 			-------#

def SetupAxis(self, temp_loc_first, temp_loc_last):
	vec = (self.matrix.to_3x3().inverted() * (temp_loc_last - temp_loc_first).normalized())

	if abs(vec[0]) > abs(vec[1]) and abs(vec[0]) > abs(vec[2]):
		return 'x'
	elif abs(vec[1]) > abs(vec[0]) and abs(vec[1]) > abs(vec[2]):
		return 'y'
	elif abs(vec[2]) > abs(vec[0]) and abs(vec[2]) > abs(vec[1]):
		return 'z'
	


def Zoom(self, context):
	ar = None
	for i in bpy.context.window.screen.areas:
		if i.type == 'VIEW_3D': ar = i
	ar = ar.spaces[0].region_3d.view_distance
	return ar

class AdvancedMove(Operator):
	''' Advanced move '''
	bl_idname = "view3d.advancedmove"
	bl_label = "Advanced Move"
	bl_options = {'REGISTER', 'UNDO'}

	def __init__(self):
		#--------Ппеременые для дхраниея координат курсора--------#
		self.temp_loc_first = None
		self.temp_loc_last = None

		#--------дает время для более точного определения выбора оси--------#
		self.count_step = 0

		# --------что бы дважды не нажимать левую кнопку--------#
		self.onlyAxis = True

		# --------Хранит пользовательскую ориентацию--------#
		self.user_orientation = None

		# --------Хранит пользовательскую ориентацию--------#
		self.axsis = None

		# --------мод инструмента--------#
		self.mode = None

		# --------список для отправки--------#

		#---------Основная матрица-----------#
		self.matrix = None
		
		#----------Матрица для отрисовки плейна----------#
		self.g_matrix = None
		
		# ----------Ось для исключения----------#
		self.exc_axis = None
		self.axis= None

		self.buffer = {'tool':'move','mode': '1', 'axis': '1'}
		self.LB = False
		self.MB = False
		self.RB = False
		self.SPACE = False
		self.mode=None

	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"

	def modal(self, context, event):
		context.area.tag_redraw()
		CreateMatrixByView(self, context, event)
		
# -----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------------#
		
		if event.type == 'LEFTMOUSE' or self.LB:
			if self.temp_loc_first is None:
				self.temp_loc_first = GetCoordMouse(self, context, event)
			self.LB = True
			if event.value == 'PRESS':
				#self.onlyAxis = False
				if self.count_step <= 2:
					self.count_step += 1
					return {'RUNNING_MODAL'}
				else:
					bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
					self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					SetConstarin.SetMoveOnly(self, context, self.axis)
					return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				bpy.context.space_data.transform_orientation = self.user_orientation
				DeleteOrientation(self, context)
				return {'FINISHED'}
		
# -----------------------RIGHT_MOUSE Exlude Axis-------------------------------------------------------------#
		
		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True
				SetConstarin.SetMoveExclude(self, context, self.exc_axis)
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = self.user_orientation
				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
				return {'FINISHED'}
				
# -----------------------MIDDLE_MOUSE No Constrain-------------------------------------------------------------#
		
		elif event.type == 'MIDDLEMOUSE' or self.MB:
			self.MB = True
			if event.value == 'PRESS':
				SetConstarin.SetMoveNoConstrain(self, context)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
			
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}
				
# -----------------------SPACEBAR Bottom			-------------------------------------------------------------#
		
		elif event.type == 'SPACE' or self.SPACE:
			self.SPACE = True
			if event.value == 'PRESS':
				SnapMoveOrientation(self, context)
				SetConstarin.SetMoveNoConstrain(self, context)
				return {'RUNNING_MODAL'}
			
			elif event.value == 'RELEASE':
				UserPresets(self, context, False)
				SetUserSnap(self, context)
				return {'FINISHED'}
			
		if event.type == 'ESC':
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		if context.space_data.type == 'VIEW_3D':
			self.user_orientation = SaveOrientation(self, context)
			GetUserSnap(self, context)
			UserPresets(self, context, True)
			
			if context.mode == "EDIT_MESH":
				# ob = bpy.context.active_object
				# count = len(ob.data.vertices)
				# sel = np.zeros(count, dtype=np.bool)
				# ob.data.vertices.foreach_get('select', sel)
				# if True in sel:
					temp = context.scene.cursor_location.copy()
					bpy.ops.view3d.snap_cursor_to_selected()
					self.center = context.scene.cursor_location.copy()
					context.scene.cursor_location = temp
					
					if self.user_orientation == 'GLOBAL':
						pass
					elif self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
						bpy.ops.transform.translate('INVOKE_DEFAULT')
					elif self.user_orientation == 'LOCAL':
						self.matrix = bpy.context.active_object.matrix_world
					elif self.user_orientation == 'NORMAL':
						self.matrix = CreateOrientation(self, context)
					else:
						self.matrix = Matrix.Translation(self.center) * context.scene.orientations[self.user_orientation].matrix.to_4x4().copy()
				# else:
				# 	self.report({'WARNING'}, "Not select elements")
				# 	return {'CANCELLED'}
			else:
				if context.active_object:
					pass
				else:
					if len(bpy.context.selected_objects):
						bpy.context.scene.objects.active = bpy.context.selected_objects[0]
					else:
						self.report({'WARNING'}, "Not select objects")
						return {'CANCELLED'}
					
				self.center = bpy.context.active_object.matrix_world * (1 / 8 * sum((Vector(b) for b in bpy.context.active_object.bound_box), Vector()))
				if self.user_orientation == 'GLOBAL':
					pass
				elif self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
					bpy.ops.transform.translate('INVOKE_DEFAULT')
					return {'FINISHED'}
				elif self.user_orientation == 'LOCAL' or self.user_orientation == 'NORMAL':
					self.matrix = context.active_object.matrix_world
				else:
					self.matrix = Matrix.Translation(self.center) * context.scene.orientations[self.user_orientation].matrix.to_4x4().copy()
				
				
			CreateMatrixByView(self, context, event)
			

			args = (self, context)
			self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_VIEW')
			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
			return {'CANCELLED'}
		

class AdvancedScale(Operator):
	''' Advanced Scale '''
	bl_idname = "view3d.advancedscale"
	bl_label = "Advanced Scale"
	bl_options = {'REGISTER', 'UNDO'}
	
	def __init__(self):
		# --------Ппеременые для дхраниея координат курсора--------#
		self.temp_loc_first = None
		self.temp_loc_last = None
		
		# --------дает время для более точного определения выбора оси--------#
		self.count_step = 0
		
		# --------что бы дважды не нажимать левую кнопку--------#
		self.onlyAxis = True
		
		# --------Хранит пользовательскую ориентацию--------#
		self.user_orientation = None
		
		# --------Хранит пользовательскую ориентацию--------#
		self.axsis = None
		
		# --------мод инструмента--------#
		self.mode = None
		
		# --------список для отправки--------#
		
		# ---------Основная матрица-----------#
		self.matrix = None
		
		# ----------Матрица для отрисовки плейна----------#
		self.g_matrix = None
		
		# ----------Ось для исключения----------#
		self.exc_axis = None
		self.axis = None
		
		self.buffer = {'tool': 'move', 'mode': '1', 'axis': '1'}
		self.LB = False
		self.RB = False
		self.MB = False
		self.LB_SP = False
		self.RB_SP = False
		self.MB = False
		self.SPACE = False
		self.ALT = False
		self.SPACE_cal = True
		self.mode = None

	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"


	def modal(self, context, event):
		context.area.tag_redraw()
		CreateMatrixByView(self, context, event)
		
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
					bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
					self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					SetConstarin.SetScaleOnly(self, context, self.axis)
					return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				bpy.context.space_data.transform_orientation = self.user_orientation
				DeleteOrientation(self, context)
				
				return {'FINISHED'}

# -----------------------RIGHT_MOUSE Exlude Axis-------------------------------------------------------------#
		
		
		
		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True
				SetConstarin.SetScaleExclude(self, context, self.exc_axis)

			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = self.user_orientation
				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
				return {'FINISHED'}

# -----------------------MIDDLE_MOUSE No Constrain-------------------------------------------------------------#
		
		elif event.type == 'MIDDLEMOUSE' or self.MB:
			self.MB = True
			if event.value == 'PRESS':
				SetConstarin.SetScaleNoConstrain(self, context)
				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
			
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}

# -----------------------SPACEBAR Bottom Setup zero-------------------------------------------------------------#
		
		elif event.type == 'SPACE' or self.SPACE:
			if self.temp_loc_first is None:
				self.temp_loc_first = GetCoordMouse(self, context, event)
			self.SPACE = True
			if event.value == 'PRESS':
				# self.onlyAxis = False
				if self.count_step <= 2:
					self.count_step += 1
					return {'RUNNING_MODAL'}
				else:
					self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					print('AXIS', self.axsis)
					SetConstarin.SetScaleOnlySetZero(self, context, self.axis)
					return {'FINISHED'}

# -----------------------ALT for negative value-------------------------------------------------------------#

		elif event.type == 'BUTTON4MOUSE' or self.ALT:
			if self.temp_loc_first is None:
				self.temp_loc_first = GetCoordMouse(self, context, event)
			self.ALT = True
			if event.value == 'PRESS':
				# self.onlyAxis = False
				if self.count_step <= 2:
					self.count_step += 1
					return {'RUNNING_MODAL'}
				else:
					self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					SetConstarin.SetScaleOnlySetNegative(self, context, self.axis)
					return {'FINISHED'}
	


		if event.type == 'ESC':
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}
	
	def invoke(self, context, event):
		if context.space_data.type == 'VIEW_3D':
			self.user_orientation = SaveOrientation(self, context)
			GetUserSnap(self, context)
			UserPresets(self, context, True)
			
			if context.mode == "EDIT_MESH":
				# ob = bpy.context.active_object
				# count = len(ob.data.vertices)
				# sel = np.zeros(count, dtype=np.bool)
				# ob.data.vertices.foreach_get('select', sel)
				# if True in sel:
				temp = context.scene.cursor_location.copy()
				bpy.ops.view3d.snap_cursor_to_selected()
				self.center = context.scene.cursor_location.copy()
				context.scene.cursor_location = temp
				
				if self.user_orientation == 'GLOBAL':
					pass
				elif self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
					bpy.ops.transform.translate('INVOKE_DEFAULT')
				elif self.user_orientation == 'LOCAL':
					self.matrix = bpy.context.active_object.matrix_world
				elif self.user_orientation == 'NORMAL':
					self.matrix = CreateOrientation(self, context)
				else:
					self.matrix = Matrix.Translation(self.center) * context.scene.orientations[
						self.user_orientation].matrix.to_4x4().copy()
				# else:
				# 	self.report({'WARNING'}, "Not select elements")
				# 	return {'CANCELLED'}
			else:
				if context.active_object:
					pass
				else:
					if len(bpy.context.selected_objects):
						bpy.context.scene.objects.active = bpy.context.selected_objects[0]
					else:
						self.report({'WARNING'}, "Not select objects")
						return {'CANCELLED'}
				
				self.center = bpy.context.active_object.matrix_world * (
				1 / 8 * sum((Vector(b) for b in bpy.context.active_object.bound_box), Vector()))
				if self.user_orientation == 'GLOBAL':
					pass
				elif self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
					bpy.ops.transform.translate('INVOKE_DEFAULT')
					return {'FINISHED'}
				elif self.user_orientation == 'LOCAL' or self.user_orientation == 'NORMAL':
					self.matrix = context.active_object.matrix_world
				else:
					self.matrix = Matrix.Translation(self.center) * context.scene.orientations[
						self.user_orientation].matrix.to_4x4().copy()
			
			CreateMatrixByView(self, context, event)
			
			args = (self, context)
			self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, args, 'WINDOW', 'POST_VIEW')
			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
			return {'CANCELLED'}


class AdvancedRotation(Operator):
	''' Advanced move '''
	bl_idname = "view3d.advancedrotation"
	bl_label = "Advanced Rotation"
	bl_options = {'REGISTER', 'UNDO'}
	
	def __init__(self):
		
		# --------дает время для более точного определения выбора оси--------#
		self.count_step_positive = 15
		self.count_step_negative = -15
		self.last_rot = None
		self.temp_loc_first= None
		self.temp_loc_last = None
		# --------что бы дважды не нажимать левую кнопку--------#
		self.onlyAxis = True
		
		# --------Хранит пользовательскую ориентацию--------#
		self.user_orientation = None
		
		# --------Хранит пользовательскую ориентацию--------#
		self.axsis = None
		
		# --------мод инструмента--------#
		self.mode = None
		
		# --------список для отправки--------#
		
		self.buffer = {'tool': '2', 'mode': '1', 'axis': '1'}
		
		# --------Для снапинга--------#
		self.snap_user = None
		
		# ---------Основная матрица-----------#
		self.matrix = None
		
		# ----------Матрица для отрисовки плейна----------#
		self.g_matrix = None
		
		# ----------Ось для исключения----------#
		self.exc_axis = None
		self.axis = None
		
		self.LB = False
		self.LB_cal = True
		self.RB = False
		self.MB = False
		self.RB_cal = True
		self.SPACE = False
		
		self.first_mouse_x = IntProperty()
		self.first_value = FloatProperty()
		self.delta = 0
	
	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"
	
	def modal(self, context, event):
		context.area.header_text_set(('Time' + str(self.time - time.clock())))
		if event.type == 'LEFTMOUSE' or self.LB:
			self.temp_loc_last = GetCoordMouse(self, context, event)
			if self.temp_loc_first is None:
				self.temp_loc_first = GetCoordMouse(self, context, event)
			if event.value == 'PRESS':
				self.LB = True
				if self.LB_cal:
					self.LB_cal = False
					self.temp_loc_last = GetCoordMouse(self, context, event)
					SetConstarin.SetRotationOnly(self, context, self.exc_axis)
				
				return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				SetUserSnap(self, context)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
				
				return {'FINISHED'}
		
		
		
		elif event.type == 'RIGHTMOUSE' or self.RB:
			if self.temp_loc_first is None:
				self.temp_loc_first = GetCoordMouse(self, context, event)
			if event.value == 'PRESS':
				self.RB = True
				self.delta = (self.first_mouse_x - event.mouse_x)
				if self.delta >= 35:
					self.first_mouse_x = event.mouse_x
					self.last_rot = 'n'
					SetConstarin.SetRotationOnlyStepgNegative(self, context, self.axis)
				elif self.delta <= -35:
					self.first_mouse_x = event.mouse_x
					self.last_rot = 'p'
					SetConstarin.SetRotationOnlyStepgPositive(self, context, self.axis)
				return {'RUNNING_MODAL'}
			
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				SetUserSnap(self, context)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')

				return {'FINISHED'}
		
		if event.type == 'ESC':
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}
	
	def invoke(self, context, event):
		self.time = time.clock()
		if context.space_data.type == 'VIEW_3D':
			self.user_orientation = SaveOrientation(self, context)
			GetUserSnap(self, context)
			UserPresets(self, context, True)
			
			if context.mode == "EDIT_MESH":
				temp = context.scene.cursor_location.copy()
				bpy.ops.view3d.snap_cursor_to_selected()
				self.center = context.scene.cursor_location.copy()
				context.scene.cursor_location = temp
				
				if self.user_orientation == 'GLOBAL':
					pass
				elif self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
					bpy.ops.transform.translate('INVOKE_DEFAULT')
				elif self.user_orientation == 'LOCAL':
					self.matrix = bpy.context.active_object.matrix_world
				elif self.user_orientation == 'NORMAL':
					self.matrix = CreateOrientation(self, context)
				else:
					self.matrix = Matrix.Translation(self.center) * context.scene.orientations[
						self.user_orientation].matrix.to_4x4().copy()
			else:
				if context.active_object:
					pass
				else:
					if len(bpy.context.selected_objects):
						bpy.context.scene.objects.active = bpy.context.selected_objects[0]
					else:
						self.report({'WARNING'}, "Not select objects")
						return {'CANCELLED'}
				
				self.center = bpy.context.active_object.matrix_world * (
					1 / 8 * sum((Vector(b) for b in bpy.context.active_object.bound_box), Vector()))
				if self.user_orientation == 'GLOBAL':
					pass
				elif self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
					bpy.ops.transform.translate('INVOKE_DEFAULT')
					return {'FINISHED'}
				elif self.user_orientation == 'LOCAL' or self.user_orientation == 'NORMAL':
					self.matrix = context.active_object.matrix_world
				else:
					self.matrix = Matrix.Translation(self.center) * context.scene.orientations[
						self.user_orientation].matrix.to_4x4().copy()
			
			CreateMatrixByView(self, context, event)
			
			args = (self, context)
			self._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_rot, args, 'WINDOW', 'POST_VIEW')
			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
			return {'CANCELLED'}


#_______________Класс для выбора констрейнов по полученым аргументам--------------------------------#


class SetConstarin(Operator):
	#-----------Constrain for move-----------#

	def SetMoveOnly(self,context, axsis):
		result = None
		if self.axis == 'x':
			result = bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		elif self.axis == 'y' :
			result = bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
		elif self.axis == 'z':
			result = bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
		print(result)
		return result

	def SetMoveExclude(self, context, axis):
		if self.exc_axis == 'x':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, True))
		elif self.exc_axis == 'y':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, True))
		elif self.exc_axis == 'z':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, True, False))
		return {'FINISHED'}

	def SetMoveNoConstrain(self, context):
		bpy.ops.transform.translate('INVOKE_DEFAULT')
		return {'FINISHED'}

	# --------Constrain for rotation----------#

	def SetRotationOnly(self,context,axis):
		if self.exc_axis == 'x':
			bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		elif self.exc_axis == 'y':
			bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
		elif self.exc_axis == 'z':
			bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=(False, False, True))
		return {'FINISHED'}

	#-----------Constrain for scale-----------#

	def SetScaleOnly(self,context,axis):
		if self.axis == 'x':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		elif self.axis == 'y' :
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
	#-----------Advanced Scale Constrain-----------#

	# -----------Align to 0-----------#

	def SetScaleOnlySetZero(self,context,axis):
		if self.axis == 'x':
			bpy.ops.transform.resize(value=(0.0, 0.0, 0.0), constraint_axis=(True, False, False))
		elif self.axis == 'y' :
			bpy.ops.transform.resize(value=(0.0, 0.0, 0.0), constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.resize(value=(0.0, 0.0, 0.0), constraint_axis=(False, False, True))
		return {'FINISHED'}

	def SetScaleExcludeSetZero(self, context, axis):
		if self.axis == 'x':
			bpy.ops.transform.resize(value=(0.0, 0.0, 0.0), constraint_axis=(False, True, True))
		elif self.axis == 'y':
			bpy.ops.transform.resize(value=(0.0, 0.0, 0.0), constraint_axis=(True, False, True))
		elif self.axis == 'z':
			bpy.ops.transform.resize(value=(0.0, 0.0, 0.0), constraint_axis=(True, True, False))
		return {'FINISHED'}

	# -----------Align to -1 -----------#

	def SetScaleOnlySetNegative(self,context,axis):
		if self.axis == 'x':
			bpy.ops.transform.resize(value=(-1.0, -1.0, -1.0), constraint_axis=(True, False, False))
		elif self.axis == 'y' :
			bpy.ops.transform.resize(value=(-1.0, -1.0, -1.0), constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.resize(value=(-1.0, -1.0, -1.0), constraint_axis=(False, False, True))
		return {'FINISHED'}

	#-----------Advanced rotation Constrain-----------#
	#-----------Step rotation 45----------#52jh


	def SetRotationOnlyStepgNegative(self,context,axis):
		if self.axis == 'x':
			bpy.ops.transform.rotate(value=0.785398, constraint_axis=(True, False, False))
		elif self.axis == 'y':
			bpy.ops.transform.rotate(value=0.785398, constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.rotate(value=0.785398, constraint_axis=(False, False, True))
		return {'FINISHED'}

	def SetRotationOnlyStepgPositive(self,context,axis):
		if self.axis == 'x':
			bpy.ops.transform.rotate(value=-0.785398, constraint_axis=(True, False, False))
		elif self.axis == 'y':
			bpy.ops.transform.rotate(value=-0.785398, constraint_axis=(False, True, False))
		elif self.axis == 'z':
			bpy.ops.transform.rotate(value=-0.785398, constraint_axis=(False, False, True))
		return {'FINISHED'}

def register():
	bpy.utils.register_module(__name__)
def unregister():
	bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
	register()
