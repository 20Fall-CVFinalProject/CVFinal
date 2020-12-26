import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import signal
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
# preprocess
import json
import os

def get_direction_field(head_x,head_y,gaze_direction,gamma=1):
	head_x = int(1920 * head_x)
	head_y = int(1080 * head_y)
	field = np.zeros((224,224))
	for idx_y,row in enumerate(field):
		for idx_x,pix in enumerate(row):
			d = np.array([idx_x*8.57 - head_x, idx_y*4.82 - head_y])
			G = d/np.linalg.norm(d)
			field[idx_y,idx_x] = max(G[0] * gaze_direction[0] + G[1] * gaze_direction[1],0) ** gamma
	return field

def save_vector_as_img(field,name):
	field = field * 255
	a = Image.fromarray(field.astype('float64')).convert('RGB')
	a.save(name)


ROOT_PATH = 'C:/Users/JQ/Documents/Code/data/'

# os.chdir(ROOT_PATH)

with open(ROOT_PATH + 'SVIP_annotation.json') as f:
	ann = json.load(f)




img_num = len(ann)
print(img_num)

for i in tqdm(range(img_num)):
	annotation = ann[i]			
	# path / x_init / y_init / w / h / gaze_point / gaze_direction / head_position
	head_x,head_y = annotation['head_position']
	gaze_direction = annotation['gaze_direction']

	gdf = get_direction_field(head_x,head_y,gaze_direction,1)
	gdf2 = get_direction_field(head_x,head_y,gaze_direction,2)
	gdf5 = get_direction_field(head_x,head_y,gaze_direction,5)

	save_vector_as_img(gdf, ROOT_PATH + annotation['path'][:-4] + '_1.png')
	save_vector_as_img(gdf2, ROOT_PATH + annotation['path'][:-4] + '_2.png')
	save_vector_as_img(gdf5, ROOT_PATH + annotation['path'][:-4] + '_5.png')

	# print(ROOT_PATH + annotation['path'])


	