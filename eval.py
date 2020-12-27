#used for evlauating the result of network
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import json
import utils
import fpn
from scipy import signal
from fpn import FPN
from GazeDirectionPathway import GazeDirectionNet

image_transforms = transforms.Compose([transforms.Resize((224, 224)),
									   transforms.ToTensor()])

head_transforms = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
gdf_tansforms = transforms.Compose([transforms.ToTensor()])

class SVIPDataset(Dataset):
	def __init__(self, root_dir, ann_file):
		self.root_dir = root_dir
		with open(root_dir + ann_file) as f:
			self.anns = json.load(f)
		self.image_num = len(self.anns)
	def __len__(self):
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
		#gaze position 
		gaze = ann['gaze_point']

		sample = {'image':image,
				  'path': ann['path'],
				  'head_image':head_image,
				  'head_position':torch.FloatTensor(head_pos),
                  'gaze_positon': gaze,
                  'eye':head_pos}
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

def get_direction_field(head_x,head_y,gaze_direction,gamma=1):
	head_x = int(1920 * head_x)
	head_y = int(1080 * head_y)
	gaze_direction = gaze_direction
	field = np.zeros((224,224))
	for idx_y,row in enumerate(field):
		for idx_x,pix in enumerate(row):
			d = np.array([idx_x*8.57 - head_x, idx_y*4.82 - head_y])
			G = d/np.linalg.norm(d)

			field[idx_y,idx_x] = max(G[0] * gaze_direction[0] + G[1] * gaze_direction[1],0) ** gamma
	field = field * 255
	a = Image.fromarray(field.astype('float64')).convert('RGB')
	# a.show()
	a.save('result/tmp/tmp.png')
	r = cv2.imread('result/tmp/tmp.png',0)
	return gdf_tansforms(r)

	
def extract_gaze_point(heatmap):
	'''extract gaze point from heatmap
	'''
	h,w = heatmap.shape
	h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
	return (w_index/w, h_index/h)

def draw_input(image,eye,i):
	x1, y1 = eye

	image_height, image_width = image.shape[:2]

	x1, y1 = image_width * x1, y1 * image_height

	x1, y1 = map(int, [x1, y1])
	cv2.circle(image, (x1, y1), 40, [0, 255, 0], 4) #eye - green

	cv2.imwrite('result/{}-input.png'.format(i), image)
	return image

def draw_direction(image,eye,direction,i):
	x1, y1 = eye
	image_height, image_width = image.shape[:2]

	x1, y1 = image_width * x1, y1 * image_height

	x2 = x1 + direction[0]*1920
	y2 = y1 + direction[1]*1920
	x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
	cv2.circle(image, (x1, y1), 40, [0, 255, 0], 4) #eye - green
	cv2.line(image, (x1, y1), (x2, y2), [255, 0, 0], 5) #line - blue
	cv2.imwrite('result/{}-direction.png'.format(i), image)

def draw_heatmap(image,heatmap,i):
	image_height, image_width = image.shape[:2]
	heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
	heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
	heatmap = cv2.resize(heatmap, (image_width, image_height))

	r = (0.8 * heatmap.astype(np.float32) + 0.2 * image.astype(np.float32)).astype(np.uint8)
	cv2.imwrite('result/{}-heatmap.png'.format(i), r)
	return r

def draw_final(image,eye,gaze_positon,heatmap,i):
	image_height, image_width = image.shape[:2]
	x1, y1 = eye
	x2, y2 = gaze_positon
	x3, y3 = extract_gaze_point(heatmap)

	x1, y1 = image_width * x1, y1 * image_height
	x2, y2 = image_width * x2, y2 * image_height
	x3, y3 = image_width * x3, y3 * image_height
	x1, y1, x2, y2, x3, y3 = map(int, [x1, y1, x2, y2, x3, y3])
	cv2.circle(image, (x1, y1), 20, [0, 255, 0], 5)
	cv2.circle(image, (x2, y2), 20, [0, 0, 255], 5)
	cv2.circle(image, (x3, y3), 20, [255, 0, 0], 5)
	cv2.line(image, (x1, y1), (x2, y2), [0, 0, 255], 5)
	cv2.line(image, (x1, y1), (x3, y3), [255, 0, 0], 5)

	heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
	heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
	heatmap = cv2.resize(heatmap, (image_width, image_height))

	r = (0.7 * heatmap.astype(np.float32) + 0.3 * image.astype(np.float32)).astype(np.uint8)
	cv2.imwrite('result/{}-final.png'.format(i), r)
	return r

def draw_temp(image,eye,gaze_positon,heatmap,i):
	image_height, image_width = image.shape[:2]
	x1, y1 = eye
	x2, y2 = gaze_positon
	x3, y3 = extract_gaze_point(heatmap)

	x1, y1 = image_width * x1, y1 * image_height
	x2, y2 = image_width * x2, y2 * image_height
	x3, y3 = image_width * x3, y3 * image_height
	x1, y1, x2, y2, x3, y3 = map(int, [x1, y1, x2, y2, x3, y3])
	cv2.circle(image, (x1, y1), 20, [0, 255, 0], 5)
	cv2.circle(image, (x2, y2), 20, [0, 0, 255], 5)
	cv2.circle(image, (x3, y3), 20, [255, 0, 0], 5)
	cv2.line(image, (x1, y1), (x2, y2), [0, 0, 255], 5)
	cv2.line(image, (x1, y1), (x3, y3), [255, 0, 0], 5)
	cv2.imwrite('result/{}-temp.png'.format(i), image)
	return image

def transform_direction(output_direction):
	output = output_direction.detach().numpy()
	return output

def draw_full(image,eye,heatmap):
	image_height, image_width = image.shape[:2]
	x1, y1 = eye
	x3, y3 = extract_gaze_point(heatmap)
	x1, y1 = image_width * x1, y1 * image_height
	x3, y3 = image_width * x3, y3 * image_height
	x1, y1, x3, y3 = map(int, [x1, y1, x3, y3])
	cv2.circle(image, (x1, y1), 15, [0, 255, 0], 5)
	cv2.circle(image, (x3, y3), 15, [255, 0, 0], 5)
	cv2.line(image, (x1, y1), (x3, y3), [255, 0, 0], 5)
	return image
def main():
	DATA_ROOT = 'data/wbq/'

	GDnet = GazeDirectionNet()
	GDnet.load_state_dict(torch.load('train3.pth',map_location=torch.device('cpu')))
	GDnet.eval()

	fpn_net = FPN()
	# fpn_net.load_state_dict(torch.load('fpn_net.pth',map_location=torch.device('cpu')))
	# fpn_net.eval()

	pretrained_dict = torch.load('fpn_net.pth',map_location=torch.device('cpu'))
	model_dict = fpn_net.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	fpn_net.load_state_dict(model_dict)

	eval_set = SVIPDataset(root_dir=DATA_ROOT,ann_file='wbq_annotation.json')
	data_loader = DataLoader(eval_set,batch_size=1,shuffle=False,num_workers=1)

	# full = cv2.imread(DATA_ROOT + 'dinner.png')
	for i, data in tqdm(enumerate(data_loader)):
		eye = data['eye']
		gaze_GT = data['gaze_positon']

		GD_input = [data['head_image'], data['head_position']]
		output_direction = GDnet(GD_input)
		direction = transform_direction(output_direction[0])
		gdf_1 = get_direction_field(eye[0],eye[1],direction,1)
		gdf_2 = get_direction_field(eye[0],eye[1],direction,2)
		gdf_5 = get_direction_field(eye[0],eye[1],direction,5)
		fpn_input = torch.cat([data['image'][0],gdf_1,gdf_2,gdf_5]).unsqueeze(0)
		output_heatmap = fpn_net(fpn_input)[0][0]

		img = cv2.imread(DATA_ROOT + data['path'][0])
		draw_temp(img,eye,gaze_GT,output_heatmap.detach().numpy(),'wbq-'+str(i))
		# draw_full(full,eye,output_heatmap.detach().numpy())
		# draw_input(img,eye,str(i))
		# img = cv2.imread(DATA_ROOT + data['path'][0])
		# draw_direction(img,eye,output_direction[0],str(i))
		img = cv2.imread(DATA_ROOT + data['path'][0])
		draw_heatmap(img,output_heatmap.detach().numpy(),'wbq-'+str(i))
		# img = cv2.imread(DATA_ROOT + data['path'][0])
		# draw_final(img,eye,gaze_GT,output_heatmap.detach().numpy(),str(i))
	# cv2.imwrite('result/dinner-full.png', full)


if __name__ == '__main__':
	main()




