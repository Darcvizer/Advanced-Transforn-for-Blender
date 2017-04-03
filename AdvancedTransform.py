bl_info = {
	"name": "Advanced Transform :)",
	"location": "View3D > Add > Object > Transform,",
	"description": "Auto set axis constrain",
	"author": "Vladislav Kindushov",
	"version": (0, 1),
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




#--------------------------Утилиты для получения координат-----------------------------------------#
#--------------------------------------------------------------------------------------------------#

#-------		For fix looping	-----------#

def UserPresets(self, context, chen):
	global save_user_drag
	#bpy_p1athy.app.binar

	if chen == True:
		save_user_drag = context.user_preferences.edit.use_drag_immediately
		if save_user_drag != True:
			context.user_preferences.edit.use_drag_immediately = True

	elif chen == False:
		context.user_preferences.edit.use_drag_immediately = save_user_drag

#-------		Get 3D world coordinate mouse position	-----------#

def GetCoordMouse(self, context, event):
	'''Get Coordinate Mouse in 3d view'''
	#scene = context.scene
	region = context.region
	rv3d = context.region_data
	coord = event.mouse_region_x, event.mouse_region_y
	#rv3d.view_rotation * Vector((0.0, 0.0, -1.0))
	view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
	loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, view_vector)
	return loc

#-------		Get 3D world coordinate mouse position		-------#

def GetSelfCoordMouse(self, context, event):
	'''Get User Coordinate Mouse in 3d view'''
	#scene = context.scene
	region = context.region
	rv3d = context.region_data

	coord = event.mouse_region_x, event.mouse_region_y
	if context.space_data.transform_orientation == 'LOCAL':
		user_matrix = GetObjMatrix(self, context)
	else:
		user_matrix = CreateOrientation(self, context)

	view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
	loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, view_vector)
	loc = user_matrix.inverted() * loc
	return loc

#--------			Get matrix object			NOT USE		-------#

def GetObjMatrix(self, context):
	act_obj = context.active_object
	matrix_obj = act_obj.matrix_world
	return matrix_obj

# --------			Create transform orientation			-------#

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

#--------			Delete self addon orientation 			-------#

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

# --------			Get axis for only transform				-------#

def SetupAxis(self, temp_loc_first, temp_loc_last):

	first_x = temp_loc_first[0]
	first_y = temp_loc_first[1]
	first_z = temp_loc_first[2]

	last_x = temp_loc_last[0]
	last_y = temp_loc_last[1]
	last_z = temp_loc_last[2]

	x = first_x - last_x
	if x < 0:
		x = -x
	y = first_y - last_y
	if y < 0:
		y = -y
	z = first_z - last_z
	if z < 0:
		z = -z

	if x > y and x > z:
		return 'x'
	elif y > x and y > z:
		return 'y'
	elif z > x and z > y:
		return 'z'

# --------			Save user transform orientation			-------#

def SaveOrientation(self, context):
	global user_orient
	user_orient = bpy.context.space_data.transform_orientation
	return user_orient

# --------			Set user transform orientation			-------#

def SetOrientation(self, context, user_orient):
	print(user_orient)
	context.space_data.transform_orientation = user_orient

# --------			Get global camera direction 			-------#

def GlobalVectorFallowView(self, context, event):
	#scene = context.scene
	region = context.region
	rv3d = context.region_data
	coord = event.mouse_region_x, event.mouse_region_y
	view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
	return rv3d.view_rotation * Vector((0.0, 0.0, -1.0))

# --------			Get user camera direction 				-------#

def SelfVectorFallowView(self, context, event):
	#scene = context.scene
	region = context.region
	rv3d = context.region_data
	coord = event.mouse_region_x, event.mouse_region_y
	view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
	view_vector = rv3d.view_rotation * Vector((-0.0, 0.0, 0.5))
	user_matrix = CreateOrientation(self, context)
	user_matrix = user_matrix.inverted() * view_vector
	return user_matrix

# --------			Get axis for Exclude transform			-------#

def ExcludeAxis(self, context, vector):
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
		return 'x'
	elif y > x and y > z:
		return 'y'
	elif z > x and z > y:
		return 'z'

#--------- 				Get user snap setting				-------#

def GetUserSnap(self, context):
	global snap_user
	# if context.scene.tool_settings.snap_element == ('VERTEX'):
	# 	snap_user = 'VERTEX'
	# elif context.scene.tool_settings.snap_element == ('EDGE'):
	# 	snap_user = 'EDGE'
	# elif context.scene.tool_settings.snap_element == ('FACE'):
	# 	snap_user = 'FACE'
	# elif context.scene.tool_settings.snap_element == ('VOLUME'):
	# 	snap_user = 'VOLUME'
	# else:
	# 	snap_user = 'INCREMENT'
	global user_snap_element
	global user_snap_target
	global user_snap_rot
	global user_snap_project
	global user_snap

	user_snap_element = context.scene.tool_settings.snap_element
	user_snap_target = 	context.scene.tool_settings.snap_target
	user_snap_rot =		context.scene.tool_settings.use_snap_align_rotation
	user_snap_project = context.scene.tool_settings.use_snap_project
	user_snap = 		context.scene.tool_settings.use_snap

#--------- 				Set user snap setting				-------#

def SetUserSnap(self, context):
	context.scene.tool_settings.snap_element =				user_snap_element
	context.scene.tool_settings.snap_target =				user_snap_target
	context.scene.tool_settings.use_snap_align_rotation = 	user_snap_rot
	context.scene.tool_settings.use_snap_project =			user_snap_project
	context.scene.tool_settings.use_snap = 					user_snap

# --------- 			Set snap setting  for move			-------#

def SnapMoveOrientation(self, context):
	context.scene.tool_settings.snap_element = 				'FACE'
	context.scene.tool_settings.snap_target = 				'CENTER'
	context.scene.tool_settings.use_snap_align_rotation = 	True
	context.scene.tool_settings.use_snap_project = 			True
	context.scene.tool_settings.use_snap = 					True



#----------------------------------------Class move------------------------------------------------#
#--------------------------------------------------------------------------------------------------#

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

		self.buffer = {'tool':'move','mode': '1', 'axis': '1'}
		self.LB = False
		self.RB = False
		self.MB = False
		self.SPACE = False
		self.mode = None

	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"

	def GiveOptions(self):
		return self.buffer



	def modal(self, context, event):
#-----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------------#

		if event.type == 'LEFTMOUSE' or self.LB:
			self.LB = True
			if event.value == 'PRESS':
				#self.onlyAxis = False
				if self.count_step <= 8:
					self.count_step += 1
					print(self.count_step)
					return {'RUNNING_MODAL'}
				else:
					if self.user_orientation != 'GLOBAL':
						self.temp_loc_last = GetSelfCoordMouse(self, context, event)
					else:
						self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					SetConstarin.SetMoveOnly(self, context, self.axis)
					bpy.context.space_data.transform_orientation = user_orient
					return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}
				#elif event.typwe == 'RIGHTMOUSE':
				#elif event.type == 'MIDLEMOUSE':

#-----------------------RIGHT_MOUSE Exlude Axis-------------------------------------------------------------#

		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True


				if self.user_orientation == 'GLOBAL':
					vector = GlobalVectorFallowView(self, context, event)
				else:
					vector = SelfVectorFallowView(self, context, event)

				self.axis = ExcludeAxis(self, context, vector)
				SetConstarin.SetMoveExclude(self, context, self.axis)

			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}

#-----------------------MIDDLE_MOUSE No Constrain-------------------------------------------------------------#

		elif event.type == 'MIDDLEMOUSE' or self.MB:
			self.MB = True
			print("chika 1")
			if event.value == 'PRESS':
				print("chika 2")
				SetConstarin.SetMoveNoConstrain(self, context)

			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}

#-----------------------SPACEBAR Bottom			-------------------------------------------------------------#

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

			if self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
				bpy.ops.transform.translate('INVOKE_DEFAULT')
				return {'FINISHED'}

			GetUserSnap(self, context)
			UserPresets(self, context, True)



			if self.user_orientation != 'GLOBAL':
				self.temp_loc_first = GetSelfCoordMouse(self, context, event)
			else:
				self.temp_loc_first = GetCoordMouse(self, context, event)

				print("podlec")
			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
			return {'CANCELLED'}

# --------------------------------------Class rotation----------------------------------------------#
# --------------------------------------------------------------------------------------------------#

class AdvancedRotation(Operator):
	''' Advanced move '''
	bl_idname = "view3d.advancedrotation"
	bl_label = "Advanced Rotation"
	bl_options = {'REGISTER', 'UNDO'}

	def __init__(self):


		#--------дает время для более точного определения выбора оси--------#
		self.count_step_positive = 15
		self.count_step_negative = -15
		self.last_rot = None
		# --------что бы дважды не нажимать левую кнопку--------#
		self.onlyAxis = True

		# --------Хранит пользовательскую ориентацию--------#
		self.user_orientation = None

		# --------Хранит пользовательскую ориентацию--------#
		self.axsis = None

		# --------мод инструмента--------#
		self.mode = None

		# --------список для отправки--------#

		self.buffer = {'tool':'2','mode': '1', 'axis': '1'}

		# --------Для снапинга--------#
		self.snap_user = None

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
		if event.type == 'LEFTMOUSE' or self.LB:
			if event.value == 'PRESS':
				self.LB = True



				if self.user_orientation == 'GLOBAL':
					vector = GlobalVectorFallowView(self, context, event)
				else:
					vector = SelfVectorFallowView(self, context, event)

				self.axis = ExcludeAxis(self, context, vector)

				if self.LB_cal:
					self.LB_cal = False
					SetConstarin.SetRotationOnly(self, context, self.axis)

				return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				SetUserSnap(self, context)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}



		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True

				if self.user_orientation == 'GLOBAL':
					vector = GlobalVectorFallowView(self, context, event)
				else:
					vector = SelfVectorFallowView(self, context, event)

				self.axis = ExcludeAxis(self, context, vector)

				self.delta = (self.first_mouse_x - event.mouse_x)
				print(self.delta)



				# if self.last_rot == 0 and self.last_rot != None:
				# 	if self.last_rot == 'n':
				# 		SetConstarin.SetRotationOnlyStepgNegative(self, context, self.axis)
				# 	elif self.last_rot == 'p':
				# 		SetConstarin.SetRotationOnlyStepgPositive(self, context, self.axis)

				if self.delta >= 35:
					#self.count_step_positive = self.delta
					#self.first_mouse_x = event.mouse_x
					#self.delta = self.first_mouse_x
					self.first_mouse_x = event.mouse_x
					self.last_rot = 'n'
					SetConstarin.SetRotationOnlyStepgNegative(self, context, self.axis)
					print('Ferst - ',self.delta, 'two - ',self.first_mouse_x  )
				elif self.delta <= -35:
					#self.count_step_negative = self.delta - self.delta - self.delta
					#self.first_mouse_x = event.mouse_x
					#self.delta = self.first_mouse_x
					self.first_mouse_x = event.mouse_x
					self.last_rot = 'p'
					SetConstarin.SetRotationOnlyStepgPositive(self, context, self.axis)


					print('thre - ', self.delta, 'for - ', self.first_mouse_x)

				return {'RUNNING_MODAL'}



			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				SetUserSnap(self, context)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}

		if event.type == 'ESC':
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		if context.space_data.type == 'VIEW_3D':
			self.user_orientation = SaveOrientation(self, context)

			if self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
				bpy.ops.transform.rotate('INVOKE_DEFAULT')
				return {'FINISHED'}

			UserPresets(self, context, True)
			GetUserSnap(self, context)
			context.scene.tool_settings.snap_element = 'INCREMENT'


			#
			# if self.user_orientation == 'GLOBAL':
			# 	vector = GlobalVectorFallowView(self, context, event)
			# else:
			# 	vector = SelfVectorFallowView(self, context, event)
			# print(self.user_orientation)
			# self.axis = ExcludeAxis(self, context, vector)
			#
			# SetConstarin.SetRotationOnly(self, context, self.axis)
			#
			# SetUserSnap(self, context)
			#
			# DeleteOrientation(self, context)
			# bpy.context.space_data.transform_orientation = user_orient
			self.first_mouse_x = event.mouse_x
			context.window_manager.modal_handler_add(self)
			return {'RUNNING_MODAL'}
		else:
			self.report({'WARNING'}, "Active space must be a View3d")
			return {'CANCELLED'}

# --------------------------------------Class Scale----------------------------------------------#
# --------------------------------------------------------------------------------------------------#

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
# -----------------------LEFT_MOUSE Only Axis Move-------------------------------------------------------------#

		if event.type == 'LEFTMOUSE' or self.LB:
			self.LB = True
			if event.value == 'PRESS':
				# self.onlyAxis = False
				if self.count_step <= 8:
					self.count_step += 1
					print(self.count_step)
					return {'RUNNING_MODAL'}
				else:
					if self.user_orientation != 'GLOBAL':
						self.temp_loc_last = GetSelfCoordMouse(self, context, event)
					else:
						self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					SetConstarin.SetScaleOnly(self, context, self.axis)

					if event.type == 'SPACE':# and event.value == 'PRESS':
						SetConstarin.SetScaleOnlySetZero(self, context, self.axis)
						print("ZERO")

					bpy.context.space_data.transform_orientation = user_orient
					return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}

# -----------------------RIGHT_MOUSE Exlude Axis-------------------------------------------------------------#

		elif event.type == 'RIGHTMOUSE' or self.RB:
			if event.value == 'PRESS':
				self.RB = True
				if self.user_orientation == 'GLOBAL':
					vector = GlobalVectorFallowView(self, context, event)
				else:
					vector = SelfVectorFallowView(self, context, event)

				self.axis = ExcludeAxis(self, context, vector)
				SetConstarin.SetScaleExclude(self, context, self.axis)

			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}

# -----------------------MIDDLE_MOUSE No Constrain-------------------------------------------------------------#

		elif event.type == 'MIDDLEMOUSE' or self.MB:
			self.MB = True
			print("chika 1")
			if event.value == 'PRESS':
				print("chika 2")
				SetConstarin.SetScaleNoConstrain(self, context)

			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}

# -----------------------SPACEBAR Bottom Setup zero-------------------------------------------------------------#

		elif event.type == 'SPACE' or self.SPACE:
			self.SPACE = True
			if event.value == 'PRESS':
				if self.count_step <= 8:
					self.count_step += 1
					print(self.count_step)
					return {'RUNNING_MODAL'}
				else:
					if self.user_orientation != 'GLOBAL':
						self.temp_loc_last = GetSelfCoordMouse(self, context, event)
					else:
						self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					#SetConstarin.SetScaleOnly(self, context, self.axis)


					if self.SPACE_cal:
						self.SPACE_cal = False
						SetConstarin.SetScaleOnlySetZero(self, context, self.axis)
					bpy.context.space_data.transform_orientation = user_orient
					return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}

# -----------------------ALT for negative value-------------------------------------------------------------#

		elif event.type == 'BUTTON4MOUSE' or self.ALT:
			self.ALT = True
			if event.value == 'PRESS':
				if self.count_step <= 8:
					self.count_step += 1
					print(self.count_step)
					return {'RUNNING_MODAL'}
				else:
					if self.user_orientation != 'GLOBAL':
						self.temp_loc_last = GetSelfCoordMouse(self, context, event)
					else:
						self.temp_loc_last = GetCoordMouse(self, context, event)
					self.axis = SetupAxis(self, self.temp_loc_first, self.temp_loc_last)
					self.ALT = False
					if self.ALT == False:
						SetConstarin.SetScaleOnlySetNegative(self, context, self.axis)

					bpy.context.space_data.transform_orientation = user_orient
					return {'RUNNING_MODAL'}
			if event.value == 'RELEASE':
				UserPresets(self, context, False)
				DeleteOrientation(self, context)
				bpy.context.space_data.transform_orientation = user_orient
				return {'FINISHED'}


		if event.type == 'ESC':
			return {'CANCELLED'}
		return {'RUNNING_MODAL'}

	def invoke(self, context, event):
		if context.space_data.type == 'VIEW_3D':

			self.user_orientation = SaveOrientation(self, context)
			if self.user_orientation == 'VIEW' or self.user_orientation == 'GIMBAL':
				bpy.ops.transform.resize('INVOKE_DEFAULT')
				return {'FINISHED'}

			GetUserSnap(self, context)
			UserPresets(self, context, True)



			if self.user_orientation != 'GLOBAL':
				self.temp_loc_first = GetSelfCoordMouse(self, context, event)
			else:
				self.temp_loc_first = GetCoordMouse(self, context, event)

				print("podlec")
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
		if self.axis == 'x':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(False, True, True))
		elif self.axis == 'y':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, False, True))
		elif self.axis == 'z':
			bpy.ops.transform.translate('INVOKE_DEFAULT', constraint_axis=(True, True, False))
		return {'FINISHED'}

	def SetMoveNoConstrain(self, context):
		bpy.ops.transform.translate('INVOKE_DEFAULT')
		return {'FINISHED'}

	# --------Constrain for rotation----------#

	def SetRotationOnly(self,context,axis):
		if self.axis == 'x':
			bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=(True, False, False))
		elif self.axis == 'y':
			bpy.ops.transform.rotate('INVOKE_DEFAULT', constraint_axis=(False, True, False))
		elif self.axis == 'z':
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
		if self.axis == 'x':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(False, True, True))
		elif self.axis == 'y':
			bpy.ops.transform.resize('INVOKE_DEFAULT', constraint_axis=(True, False, True))
		elif self.axis == 'z':
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
	#-----------Step rotation 45----------#


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


#----------------------------------Class Clear orientation for macro--------------------------------#

class Clear(Operator):
	bl_idname = "view3d.clear"
	bl_label = "Clear Orintation"

	@classmethod
	def poll(cls, context):
		return context.space_data.type == "VIEW_3D"

	def execute(self, context):
		DeleteOrientation(self, context)
		SetOrientation(self, context, user_orient)
		#print("CLEAR")
		return {'FINISHED'}

# ----------------------------------Class macro-----------------------------------------#

class TransformMacro(Macro):
	bl_idname = 'transform_macro'
	bl_label = 'Transform Macro'

	'''def execute(self, context):
		bpy.utils.register_class(AdvancedMove)
		bpy.utils.register_class(Clear)
		bpy.utils.register_class(TransformMacro)

		TransformMacro.define(view3d.advancedmove)
		TransformMacro.define(view3d.clear)
		return {'FINISHED'}'''


# class AdvancedTransform(bpy.types.AddonPreferences):
# 	bl_idname = __name__
# 	some_prop = bpy.types.KeyMapItem.type = 'NONE'
#
# 	def draw(self, context):
# 		layout = self.layout
# 		layout.prop(self, "some_prop")

# class AddonPreferences(bpy.types.AddonPreferences):
# 	bl_idname = __name__
#
# 	def draw(self, context):
# 		layout = self.layout
# 		wm = bpy.context.window_manager
# 		box = layout.box()
# 		split = box.split()
# 		col = split.column()
# 		col.label('Setup Pie menu Hotkey')
# 		col.separator()
# 		wm = bpy.context.window_manager
# 		kc = wm.keyconfigs.user
# 		km = kc.keymaps['3D View Generic']
# 		kmi = get_hotkey_entry_item(km, 'wm.call_menu_pie', 'pie.test_pie_menu')
# 		if kmi:
# 			col.context_pointer_set("keymap", km)
# 			rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)
# 		else:
# 			col.label("No hotkey entry found")
# 			col.operator(Template_Add_Hotkey.bl_idname, text="Add hotkey entry", icon='ZOOMIN')



def register():
	bpy.utils.register_module(__name__)
	kc = bpy.context.window_manager.keyconfigs.addon
	if kc:
		km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
		kmi = km.keymap_items.new(AdvancedMove.bl_idname, 'G', 'PRESS', )
		kmi = km.keymap_items.new('view3d.advancedrotation', 'R', 'PRESS', )
		kmi = km.keymap_items.new('view3d.advancedscale', 'S', 'PRESS', )


def unregister():
	bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
	register()