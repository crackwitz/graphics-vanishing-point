#!/usr/bin/env python3

import numpy as np
import cv2 as cv # for drawing

np.set_printoptions(precision=4, linewidth=140, suppress=True)

# screen:
#  x right
#  y up
#  z towards you

# world:
#  x right
#  y away
#  z up

def to_projective(matrix):
	return np.column_stack(np.broadcast(*matrix, 1))

def project(matrix):
	return np.array(matrix[0:3] / matrix[3])

def identity():
	return np.matrix(np.eye(4))

def translate(arg=None, x=None, y=None, z=None):
	if arg is not None:
		if hasattr(arg, '__len__'):
			x,y,z = arg
		else:
			x = y = z = arg

	x = x or 0
	y = y or 0
	z = z or 0

	result = np.eye(4)
	result[0:3,3] = (x, y, z)
	return np.matrix(result)

def scale(x=1, y=1, z=1):
	result = np.diag([x, y, z, 1])
	return np.matrix(result)

def rotate(vector, angle):
	R = np.eye(4)
	vector = np.array(vector) * angle/180*np.pi
	dst,jacobian = cv.Rodrigues(vector)
	R[:3,:3] = dst
	return np.matrix(R)

def crossproduct(a, b):
	return np.matrix(np.cross(a.T, b.T)).T

def normalize(v):
	return v / np.linalg.norm(v[:3])

def lookat(pfrom, pto, up):
	if isinstance(pfrom, (list, tuple)):
		pfrom = np.matrix(pfrom).T

	if isinstance(pto, (list, tuple)):
		pto = np.matrix(pto).T

	if isinstance(up, (list, tuple)):
		up = np.matrix(up).T

	# https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
	forward = normalize(pfrom - pto) # towards camera
	right = normalize(crossproduct(up, forward))
	up = normalize(crossproduct(forward, right))
	cam2world = np.matrix(np.eye(4))
	cam2world[:3,0] = right
	cam2world[:3,1] = up
	cam2world[:3,2] = forward
	cam2world[:3,3] = pfrom

	world2cam = cam2world ** -1
	return world2cam


def create_canvas(screensize):
	w,h = screensize
	canvas = np.zeros((h, w, 3), dtype=np.uint8)
	return canvas

def intshift(val, shift):
	return tuple((val * 2**shift).round().astype(np.int))

def draw_vertices(canvas, V, Vc):
	shift = 4
	for v,vc in zip(V, Vc):
		if np.isnan(v).any():
			print("isnan", v)
			continue

		#if v[2] >= 0: continue # wrong side of the image plane

		if np.isinf(v).any(): continue

		center = (
			int(v[0] * 2**shift),
			int(v[1] * 2**shift),
		)

		radius = int(3.0 * 2**shift)

		cv.circle(canvas, center, radius=radius, shift=shift, color=vc, thickness=-1, lineType=cv.LINE_AA)


def draw_edges(canvas, V, E, Ec):
	shift = 4

	for e,ec  in zip(E, Ec):
		(i0, i1) = e

		p0 = intshift(V[i0,:2], shift=shift)
		p1 = intshift(V[i1,:2], shift=shift)

		color = ec[:3] if ec  else (128, 128, 128)

		cv.line(canvas, p0, p1, shift=shift, color=color, thickness=1, lineType=cv.LINE_AA)

def draw_triangles(canvas, V, T, Tc):
	shift = 4

	for t,tc  in zip(T, Tc):
		(i0, i1, i2) = t

		p0 = intshift(V[i0,:2], shift=shift)
		p1 = intshift(V[i1,:2], shift=shift)
		p2 = intshift(V[i2,:2], shift=shift)

		color = tc[:3] if tc else (64, 64, 64)

		cv.fillPoly(canvas, np.array([[p0, p1, p2]]), shift=shift, color=color, lineType=cv.LINE_AA)

def draw_object(canvas, PV, obj):
	# Projection * View * Model * vertices

	(V,E,T,M) = obj
	V,Vc = zip(*V)

	V = PV * M * np.matrix(V).T
	V = project(V).T

	#print(V)

	if len(T) > 0:
		T,Tc = zip(*T)
		draw_triangles(canvas, V, T, Tc)

	if len(E) > 0:
		E,Ec = zip(*E)
		draw_edges(canvas, V, E, Ec)

	draw_vertices(canvas, V, Vc)

def draw_all(PV):
	canvas = create_canvas(screensize)

	for obj in objects:
		draw_object(canvas, PV, obj)

	return canvas

### objects: vertices, lines, triangles

# triad
triad = (
	[ # vertices
		( ( 0,  0,  0, 1), (255, 255, 255) ),
		( ( 1,  0,  0, 1), (  0,   0, 255) ),
		( ( 0,  1,  0, 1), (  0, 255,   0) ),
		( ( 0,  0,  1, 1), (255,   0,   0) ),

		# vanishing points
		( ( 0,  0, +1, 0), (  0,   0, 255) ),
		( ( 0,  0, -1, 0), (255, 255,   0) ),
		( ( 0, +1,  0, 0), (  0, 255,   0) ),
		( ( 0, -1,  0, 0), (255,   0, 255) ),
		( (+1,  0,  0, 0), (255,   0,   0) ),
		( (-1,  0,  0, 0), (  0, 255, 255) ),
	],
	[ # lines
		( (0, 1), (  0,   0, 255) ),
		( (0, 2), (  0, 255,   0) ),
		( (0, 3), (255,   0,   0) ),
	],
	[ # triangles
		( (0, 1, 2), (  0, 255, 255, 64) ),
		( (0, 1, 3), (255,   0, 255, 64) ),
		( (0, 2, 3), (255, 255,   0, 64) ),
	],
	# world <- object transformation
	translate()
)


# box
box = (
	[ # vertices
		( ( 0,  0,  0, 1), ( 64,  64,  64) ), # 0
		( ( 1,  0,  0, 1), (  0,   0, 255) ), # 1
		( ( 0,  1,  0, 1), (  0, 255,   0) ), # 2
		( ( 1,  1,  0, 1), (  0, 255, 255) ), # 3
		( ( 0,  0,  1, 1), (255,   0,   0) ), # 4
		( ( 1,  0,  1, 1), (255,   0, 255) ), # 5
		( ( 0,  1,  1, 1), (255, 255,   0) ), # 6
		( ( 1,  1,  1, 1), (255, 255, 255) ), # 7
	],
	[ # lines
		( (0, 1), None ),
		( (1, 3), None ),
		( (3, 2), None ),
		( (2, 0), None ),
		( (4, 5), None ),
		( (5, 7), None ),
		( (7, 6), None ),
		( (6, 4), None ),
		( (0, 4), None ),
		( (1, 5), None ),
		( (2, 6), None ),
		( (3, 7), None ),
	],
	[ # triangles
	],
	# world <- object transformation
	translate([2, 2, 2])
)


objects = [
	triad,
	box
]

screensize = (1024, 1024)
#screensize = (640, 480)

cv.namedWindow("canvas") #, cv.WINDOW_OPENGL)
#cv.resizeWindow("canvas", screensize)


uivars = {}

def create_var(varname, vmin, vmax, vstep, vdefault):
	uivars[varname] = vdefault
	globals()[varname] = vdefault
	irange = int(round((vmax-vmin) / vstep))
	idefault = int(round((vdefault - vmin) / vstep))

	def callback(newvalue):
		val = vmin + newvalue * vstep
		uivars[varname] = val
		globals()[varname] = val
		print(f"{varname} = {val:.3g}")

		render()

	cv.createTrackbar(varname, "canvas", idefault, irange, callback)
	return callback


create_var("lookat_tx", -10, +10, 0.1, 5)
create_var("lookat_ty", -10, +10, 0.1, 5)
create_var("lookat_tz", -10, +10, 0.1, 5)
#create_var("lookat_rx", -45, +45, 1, 0)
#create_var("lookat_ry", -45, +45, 1, 0)

def render():
	global points
	global Model, View, Projection

	# camera plane <- view
	Projection = np.matrix(np.eye(4))
	Projection[3,:] = 0
	Projection[3,2] = -1 # project onto z=-1, which is away from the camera plane
	#Projection[2,3] = -1
	# TODO: http://learnwebgl.brown37.net/08_projections/projections_perspective.html

	# screen plane <- camera plane
	w,h = screensize
	Screen = translate([w/2, h/2, 0]) * scale(h/2, -h/2, 1) # h/2 ~ 90 degrees
	Projection = Screen * Projection

	#import pdb; pdb.set_trace()

	# view <- world
	View = identity()
	#View *= rotate((1,0,0), lookat_rx) # tilt
	#View *= rotate((0,1,0), lookat_ry) # pan... relative to frame looking at origin
	View *= lookat(pfrom=(lookat_tx, lookat_ty, lookat_tz), pto=(1,1,1), up=(0,0,1))

	PV = Projection * View

	canvas = draw_all(PV)

	cv.imshow("canvas", canvas)


while True:
	render()
	key = cv.waitKey(1000)
	if key in (13, 27):
		break

