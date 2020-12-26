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

import fpn
from torch import nn,optim

EPOCH_NUM = 1

image_transforms = transforms.Compose([transforms.Resize((224, 224)),
									   transforms.ToTensor()])
									   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

head_transforms = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor()])
									  # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

gdf_transformer = transforms.Compose([transforms.ToTensor()])


class SVIPDataset(Dataset):
	def __init__(self, root_dir, ann_file):
		self.root_dir = root_dir
		with open(root_dir + ann_file) as f:
			self.anns = json.load(f)
		self.image_num = len(self.anns)
		# print(self.image_num)
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
		# print("IMG: ",image)
		#head image
		# head_image = self.get_head_img(img,ann)
		#head position
		head_pos = ann['head_position']
		#gaze direction
		gaze_dir = ann['gaze_direction']
		#gaze_direction_field
		gdf = cv2.imread(self.root_dir + ann['path'][:-4] + '_1.png',0)
		# gdf = cv2.fromarray(gdf)
		# gdf = Image.fromarray(gdf)
		# save_vector_as_img(gdf,'1.png')
		# gdf = cv2.imread('1.png',0)
		gdf = gdf_transformer(gdf)

		gdf_2 = cv2.imread(self.root_dir + ann['path'][:-4] + '_2.png',0)
		gdf_2 = gdf_transformer(gdf_2)

		gdf_5 = cv2.imread(self.root_dir + ann['path'][:-4] + '_5.png',0)
		gdf_5 = gdf_transformer(gdf_5)

		# print("GDF:",gdf)
		# gdf_2 = self.get_direction_field(head_pos[0],head_pos[1],gaze_dir,2)
		# gdf_5 = self.get_direction_field(head_pos[0],head_pos[1],gaze_dir,5)
		# gdf_2 = cv2.fromarray(gdf_2)
		# gdf_5 = cv2.fromarray(gdf_5)
		# gdf_2 = Image.fromarray(gdf_2)
		# save_vector_as_img(gdf_2,'2.png')
		# gdf_2 = cv2.imread('2.png',0)
		# gdf_2 = gdf_transformer(gdf_2)

		# gdf_5 = Image.fromarray(gdf_5)
		# save_vector_as_img(gdf_5,'5.png')
		# gdf_5 = cv2.imread('2.png',0)
		# gdf_5 = gdf_transformer(gdf_5)

		# show_vector_as_img(gdf)
		# show_vector_as_img(gdf_2)
		# show_vector_as_img(gdf_5)
		#gaze point
		gaze = ann['gaze_point']
		#heatmap
		heatmap = self.get_heatmap_GT(gaze[0],gaze[1])

		# print("="*30)
		# print(image.shape)
		# print(gdf.shape)
		# print("="*30)

		sample = {'image':image,
				  'head_image':0,
				  'head_position':torch.FloatTensor(head_pos),
				  'gaze_direction':torch.FloatTensor(gaze_dir),
				  'gaze_direction_field': torch.cat([image,gdf,gdf_2,gdf_5],dim=0),
				  # 'gaze_direction_field': torch.cat([image.to(torch.float64),gdf.to(torch.float64),gdf_2.to(torch.float64),gdf_5.to(torch.float64)],dim=0),
                  'gaze_positon': torch.FloatTensor(gaze),
                  'heatmap': torch.FloatTensor(heatmap).unsqueeze(0),
                  'path': ann['path']}
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
				field[idx_y,idx_x] = max(G[0] * gaze_direction[0] + G[1] * gaze_direction[1],0) ** gamma
		return field

	def get_heatmap_GT(self,gaze_x,gaze_y,kernlen=21, std=3):
		'''Get ground truth heatmap or (51,9)
		'''
		gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
		kernel_map = np.outer(gkern1d, gkern1d)

		gaze_x = int(56*gaze_x)
		gaze_y = int(56*gaze_y)
		
		k_size = kernlen // 2

		x1, y1 = gaze_x - k_size, gaze_y - k_size
		x2, y2 = gaze_x + k_size, gaze_y + k_size

		h, w = 56,56
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
		return heatmap[:56, :56]


def show_vector_as_img(vector):
	img = vector.squeeze()
	img = img.cpu()
	img = img.detach().numpy() * 255
	img = img.astype(np.uint8)
	img = Image.fromarray(img)
	img.show()


def save_vector_as_img(field,name):
	field = field * 255
	a = Image.fromarray(field.astype('float64')).convert('RGB')
	a.save(name)


def main():
	ROOT_PATH = 'D:/DataSet/data/'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Train on device: ",device)


	train_set = SVIPDataset(root_dir=ROOT_PATH,
                           ann_file='SVIP_annotation.json')

	train_data_loader = DataLoader(train_set, batch_size=1,
                                   shuffle=True, num_workers=1)

	fpn_net = fpn.FPN()
	fpn_net.to(device)

	pretrained_dict = torch.load(ROOT_PATH + 'fpn_net.pth')
	model_dict = fpn_net.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	fpn_net.load_state_dict(model_dict)

	# optimizer = optim.SGD(fpn_net.parameters(),lr=0.01)

	learning_rate = 0.000001
	optimizer = optim.Adam([{'params': fpn_net.parameters(), 'initial_lr': learning_rate}],
                           lr=learning_rate, weight_decay=0.0001)

	criterion = nn.BCELoss()

	avg_loss = 0
	counter = 0
	for epoch in range(EPOCH_NUM):
		print('='*20 + 'start training epoch',epoch,'='*20)
		for i, data in enumerate(train_data_loader):

			heatmap = data['heatmap'].to(device)
			# print('6:',heatmap)
			cat_input = data['gaze_direction_field'].to(device)
			optimizer.zero_grad()
			# print("PATH: ",data['path'])
			# print("INPUT: ",cat_input)


			output_heatmap = fpn_net(cat_input)

			# print(output_heatmap)

			# print(heatmap.dtype)
			# print(output_heatmap.dtype)
			# print("="*30)

			loss = criterion(output_heatmap,heatmap)
			
			loss.backward()
			optimizer.step()
			# print(i,loss.item())
			avg_loss += loss.item()
			counter += 1

			if i % 1000 == 0:
				avg_loss = avg_loss/counter
				print("Average Loss = ",avg_loss)
				avg_loss = 0
				counter = 0
				if i < 5000:
					show_vector_as_img(output_heatmap)
		# if epoch%10 == 0 or epoch == EPOCH_NUM-1:
		# 	print(loss.item())
		# 	show_vector_as_img(output_heatmap)
		torch.save(fpn_net.state_dict(),ROOT_PATH + 'fpn_net.pth')
	# print("Average Loss = ",avg_loss)




if __name__ == '__main__':
    main()