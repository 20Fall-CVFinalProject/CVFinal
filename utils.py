import cv2
from PIL import Image
import numpy as np
# data transform for image in svip data
# svip_data_transforms = transforms.Compose([
# 		# place holder
# 	])

# mit_data_transforms = transforms.Compose([
# 		# place holder
# 	])
def get_direction(x_0,y_0,x_1,y_1):
	'''Get normalized direction from two points 0 -> 1
	'''
	d = np.array([x_1 - x_0, y_1 - y_0])
	d = d/np.linalg.norm(d)
	return d

def get_gaze_direction_GT(head_x,head_y,gaze_x,gaze_y):
	'''Get gaze direction from head position and gaze point
	'''
	return get_direction(head_x,head_y,gaze_x,gaze_y)


def get_heatmap_GT():
	'''Get ground truth heatmap
	'''


def get_direction_field(height,width,head_x,head_y,gaze_direction,gamma=1):
	'''get direction field from head position and gaze direction
	'''
	if isinstance(head_x,float) and isinstance(head_y,float):
		head_x = int(width * head_x)
		head_y = int(height * head_y)
	field = np.zeros((height,width))
	for idx_y,row in enumerate(field):
		for idx_x,pix in enumerate(row):
			G = get_direction(head_x,head_y,idx_x,idx_y)
			field[idx_y,idx_x] = max(G[0] * gaze_direction[0] + G[1] * gaze_direction[1],0) ** gamma
	field = field * 255
	return Image.fromarray(field.astype('float64'))

def extract_gaze_point():
	'''extract gaze point drom heatmap
	'''
	pass

def MIT_get_head_img(img_path,annotation,show=0):
	'''get the head image for MIT data
	'''
	image = cv2.imread(img_path, cv2.IMREAD_COLOR)
	h,w,_ = image.shape

	y_0 = int(annotation['y_init'] * h)
	y_1 = int((annotation['y_init'] + annotation['h']) * h)
	x_0 = int(annotation['x_init'] * w)
	x_1 = int((annotation['x_init'] + annotation['w']) * w )

	face_image = image[y_0:y_1, x_0:x_1,:]
	face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
	face_image = Image.fromarray(face_image)

	if show:
		face_image.show()
		if show == 2:
			img = Image.open(img_path)
			img.show()

	return face_image

def MIT_extract_annotation_as_dict(annotation):
	'''each line of MIT transform into annotation dictionary
	'''
	f = annotation.split(',')
	ann  = {'path':f[0],
			'index':f[1],
			'x_init': float(f[2]),
			'y_init': float(f[3]),
			'w': float(f[4]),
			'h': float(f[5]),
			'eye_x': float(f[6]),
			'eye_y': float(f[7]),
			'gaze_x': float(f[8]),
			'gaze_y': float(f[9]),
			'meta': f[10]}
	return ann



