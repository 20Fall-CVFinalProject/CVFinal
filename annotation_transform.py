import json
import numpy as np
with open('data/SVIP/annotation.json') as f:
	ann = json.load(f)
path = ann['path']
boxes = ann['boxes']
points = ann['points']

def get_direction(x_0,y_0,x_1,y_1):
	d = np.array([x_1 - x_0, y_1 - y_0])
	d = d/np.linalg.norm(d)
	return d

image_num = len(boxes)
output = []
for i in range(image_num):
	p = path[i]
	x_init,y_init,x_w,y_h = boxes[i]
	gaze_x, gaze_y = points[i]
	if x_init<0 or y_init<0 or (x_init+x_w)>=1920 or (y_init+y_h)>=1080:
		continue

	gaze_direction = get_direction(x_init+x_w//2,y_init+y_h//2,gaze_x,gaze_y)

	annotation = {
		'path': p,
		'x_init': x_init/1920,
		'y_init': y_init/1080,
		'w': x_w/1920,
		'h': y_h/1080,
		'head_position': [(x_init+0.5*x_w)/1920,(y_init+0.5*y_h)/1080],
		'gaze_direction':  gaze_direction.tolist(),
		'gaze_point': [gaze_x/1920,gaze_y/1080]
	}
	output.append(annotation)

with open('data/SVIP/SVIP_annotation.json','w') as f:
	json.dump(output,f)



