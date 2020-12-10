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
image_transforms = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

head_transforms = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class MITDataset(Dataset):
	def __init__(self, root_dir, ann_file):
		self.root_dir = root_dir
		with open(root_dir + ann_file) as f:
			self.ann_list = f.read().strip('\n').split('\n')

		self.image_num = len(self.ann_list)
	def __len__(self):
		return self.image_num

	def __getitem__(self,idx):
		ann = self.extract_MIT(self.ann_list[idx])

		img = cv2.imread(self.root_dir + ann['path'], cv2.IMREAD_COLOR)

		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		image = image_transforms(image)

		head_image = self.get_head_img(img,ann)

		head_pos = [ann['eye_x'],ann['eye_y']]

		gaze_dir = self.get_direction(ann['eye_x'],ann['eye_y'],ann['gaze_x'],ann['gaze_y'])

		gaze = [ann['gaze_x'],ann['gaze_y']]

		heatmap = self.get_heatmap_GT(img.shape[0],img.shape[1],ann['gaze_x'],ann['gaze_y'])

		sample = {'image':image,
				  'head_image':head_image,
				  'head_position':torch.FloatTensor(head_pos),
				  'gaze_direction':torch.from_numpy(gaze_dir),
                  'gaze_positon': torch.FloatTensor(gaze),
                  'heatmap': torch.FloatTensor(heatmap).unsqueeze(0)}

		return sample
	def extract_MIT(self,ann_string):
		f = ann_string.split(',')
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
		if ann['x_init']<0:
			ann['x_init'] = 0
		if ann['y_init']<0:
			ann['y_init'] = 0
		if ann['x_init'] + ann['w']>1:
			ann['w'] = 1 - ann['x_init']
		if ann['y_init'] + ann['h']>1:
			ann['h'] = 1 - ann['y_init']
		return ann

	def get_head_img(self,image,annotation):
		h,w,_ = image.shape
		y_0 = int(annotation['y_init'] * h)
		y_1 = int((annotation['y_init'] + annotation['h']) * h)
		x_0 = int(annotation['x_init'] * w)
		x_1 = int((annotation['x_init'] + annotation['w']) * w )
		face_image = image[y_0:y_1, x_0:x_1,:]
		face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
		face_image = Image.fromarray(face_image)
		face_image = head_transforms(face_image)
		return face_image

	def get_direction(self,x_0,y_0,x_1,y_1):
		d = np.array([x_1 - x_0, y_1 - y_0])
		d = d/np.linalg.norm(d)
		return d

	def get_heatmap_GT(self,height,width,gaze_x,gaze_y,kernlen=21, std=3):
		'''Get ground truth heatmap or (51,9)
		'''
		gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
		kernel_map = np.outer(gkern1d, gkern1d)

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
		return heatmap

def main():
	train_set = MITDataset(root_dir='data/MIT/',
                           ann_file='train_annotations.txt')

	train_data_loader = DataLoader(train_set, batch_size=32 * 4,
                                   shuffle=True, num_workers=16)

	for i, data in enumerate(train_data_loader):
		image = data['image']
		head_image = data['head_image']
		head_position = data['head_positio']
		gaze_direction = data['gaze_direction']
		gaze_positon = data['gaze_direction']
		heatmap = heatmap['heatmap']
		print(type(data['image']))
		print(type(data['head_image']))
		print(type(data['head_positio']))
		print(type(data['gaze_direction']))
		print(type(data['gaze_direction']))
		print(type(heatmap['heatmap']))



if __name__ == '__main__':
    main()


