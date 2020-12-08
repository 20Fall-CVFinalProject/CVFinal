import cv2
from PIL import Image
import numpy as np
from scipy import signal
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


def get_heatmap_GT(height,width,gaze_x,gaze_y,kernlen=21, std=3):
	'''Get ground truth heatmap or (51,9)
	'''
	gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
	kernel_map = np.outer(gkern1d, gkern1d)
	if isinstance(gaze_x,float) and isinstance(gaze_y,float):
		gaze_x = int(width * gaze_x)
		gaze_y = int(height * gaze_y)

	k_size = kernlen // 2
	x1, x2 = gaze_x - k_size, gaze_x + k_size
	y1, y2 = gaze_y - k_size, gaze_y + k_size
	if x2 >= width:
		width = x2
	if y2 >= height:
		height = y2
	left, top, k_left, k_top = x1, y1, 0, 0
	if x1 < 0:
		left = 0
		k_left = -x1
	if y1 < 0:
		top = 0
		k_top = -y1

	heatmap = np.zeros((height,width))
	heatmap[top:y2+1, left:x2+1] = kernel_map[k_top:, k_left:]
	heatmap = heatmap * 255
	return Image.fromarray(heatmap.astype('float64')).convert('RGB')

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
	return Image.fromarray(field.astype('float64')).convert('RGB')

def extract_gaze_point(heatmap):
	'''extract gaze point drom heatmap
	'''
	h,w = heatmap.shape
	h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
	return (w_index/w, h_index/h)


def get_head_img(img_path,annotation,show=0):
	'''
	show = 0 -> don't show
	show = 1 -> show head image
	show = 2 -> show haed image and original image
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

def SVIP_extract_annotation_as_dict(path,box,point,size):
	'''generate MIT-like annotations for SVIP data
	'''
	ann = {'path':path,
			'index':None,
			'x_init': box[0]/size[1],
			'y_init': box[1]/size[0],
			'w': box[2]/size[1],
			'h': box[3]/size[0],
			'eye_x': (box[0]+box[2]/2)/size[1],
			'eye_y': (box[1]+box[3]/2)/size[0],
			'gaze_x': point[0]/size[1],
			'gaze_y': point[1]/size[0],
			'meta': None}
	return ann



