# running train
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
image_transforms = transforms.Compose([transforms.Resize((224, 224)),
									   transforms.ToTensor(),
									   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

head_transforms = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class SVIPDataset(Dataset):
	def __init__(self, root_dir, ann_file):
		self.root_dir = root_dir
		with open(root_dir + ann_file) as f:
			self.anns = json.load(f)
		self.image_num = len(self.anns)
		print(self.image_num)
	def __len__(self):
		# return 2
		return self.image_num

	def __getitem__(self,idx):
		ann = self.anns[idx]

		img = cv2.imread(self.root_dir + ann['path'], cv2.IMREAD_COLOR)
		#image
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		image = image_transforms(image)
		#head image
		head_image = self.get_head_img(img,ann)
		#head position
		head_pos = ann['head_position']
		#gaze direction
		gaze_dir = ann['gaze_direction']
		#gaze_direction_field
		gdf = self.get_direction_field(head_pos[0],head_pos[1],gaze_dir)
		gdf = torch.from_numpy(gdf).unsqueeze(0)
		gdf_2 = torch.pow(gdf,2)
		gdf_5 = torch.pow(gdf,5)
		#gaze point
		gaze = ann['gaze_point']
		#heatmap
		heatmap = self.get_heatmap_GT(gaze[0],gaze[1])

		sample = {'image':image,
				  'head_image':head_image,
				  'head_position':torch.FloatTensor(head_pos),
				  'gaze_direction':torch.FloatTensor(gaze_dir),
				  'gaze_direction_field': torch.cat([image,gdf,gdf_2,gdf_5],dim=0),
                  'gaze_positon': torch.FloatTensor(gaze),
                  'heatmap': torch.FloatTensor(heatmap).unsqueeze(0)}
		# print(sample["gaze_positon"])
		return sample

	def get_head_img(self,image,annotation):
		h,w,_ = image.shape
		x_0 = int(w*annotation['x_init'])
		x_1 = int(w*(annotation['x_init']+annotation['w']))
		y_0 = int(h*annotation['y_init'])
		y_1 = int(h*(annotation['y_init']+annotation['h']))
		face_image = image[y_0:y_1, x_0:x_1,:]
		face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
		face_image = Image.fromarray(face_image)
		face_image = head_transforms(face_image)
		return face_image

	def get_direction_field(self,head_x,head_y,gaze_direction,gamma=1):
		head_x = int(1920 * head_x)
		head_y = int(1080 * head_y)
		field = np.zeros((224,224))
		for idx_y,row in enumerate(field):
			for idx_x,pix in enumerate(row):
				d = np.array([idx_x*8.57 - head_x, idx_y*4.82 - head_y])
				G = d/np.linalg.norm(d)
				field[idx_y,idx_x] = max(G[0] * gaze_direction[0] + G[1] * gaze_direction[1],0)
		return field

	def get_heatmap_GT(self,gaze_x,gaze_y,kernlen=21, std=3):
		'''Get ground truth heatmap or (51,9)
		'''
		gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
		kernel_map = np.outer(gkern1d, gkern1d)

		gaze_x = int(224*gaze_x)
		gaze_y = int(224*gaze_y)
		
		k_size = kernlen // 2

		x1, y1 = gaze_x - k_size, gaze_y - k_size
		x2, y2 = gaze_x + k_size, gaze_y + k_size

		h, w = 224,224
		if x2 >= w:
			w = x2 + 1
		if y2 >= h:
			h = y2 + 1
		heatmap = np.zeros((h, w))
		left, top, k_left, k_top = x1, y1, 0, 0
		if x1 < 0:
			left = 0
			k_left = -x1
		if y1 < 0:
			top = 0
			k_top = -y1

		heatmap[top:y2+1, left:x2+1] = kernel_map[k_top:, k_left:]
		return heatmap[:224, :224]

def main():
	train_set = SVIPDataset(root_dir='data/SVIP/',
                           ann_file='SVIP_annotation.json')

	train_data_loader = DataLoader(train_set, batch_size=1,
                                   shuffle=False, num_workers=1)
	for i, data in tqdm(enumerate(train_data_loader)):
		



if __name__ == '__main__':
    main()


