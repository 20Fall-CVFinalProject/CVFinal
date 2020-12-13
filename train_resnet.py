import json
import utils
from GazeDirectionPathway import GazeDirectionNet 
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import os
import torch.optim as optim
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MIT_DATA_PATH = 'example_data/MIT_annotation.txt'
MIT_ORIGINAL_PATH = 'example_data/MIT_original.jpg'


def GDLoss(direction, gt_direction):
	cosine_similarity = nn.CosineSimilarity()
	gt_direction = gt_direction.unsqueeze(0)
	# print(direction, gt_direction)
	loss = torch.mean(1 - cosine_similarity(direction, gt_direction))
	return loss

with open(MIT_DATA_PATH,'r') as ann:
    data_mit = ann.read()
# print("MIT:",data_mit)
image_transforms = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

head_transforms = transforms.Compose([transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ann = utils.MIT_extract_annotation_as_dict(data_mit)
# print(ann)
hx, hy = ann['eye_x'], ann['eye_y']
# print(hx,hy)
# image = cv2.imread(MIT_ORIGINAL_PATH, cv2.IMREAD_COLOR)
# print(image)
head_img = utils.get_head_img(MIT_ORIGINAL_PATH,ann,0)
head_img = head_img.resize((224,224))
head_img = transforms.ToTensor()(head_img)
head_img = head_img.unsqueeze(0)

head_pos =  torch.Tensor([hx,hy])
head_pos = head_pos.unsqueeze(0)

direction_gt = utils.get_gaze_direction_GT(hx,hy,ann['gaze_x'],ann['gaze_y'])
direction_gt = torch.from_numpy(direction_gt)
print(direction_gt)


net = GazeDirectionNet() 
print("train")
learning_rate = 0.8
optimizer = optim.Adam([{'params': net.head_feature_net.parameters(), 
							'initial_lr': learning_rate},
							{'params': net.head_feature_process.parameters(), 
							'initial_lr': learning_rate},
							{'params': net.head_pos_net.parameters(), 
							'initial_lr': learning_rate},
							{'params': net.concatenate_net.parameters(), 
							'initial_lr': learning_rate}],
							lr=learning_rate, weight_decay=0.0001)
epoch = 300
for i in range(epoch):
	print(i)
	optimizer.zero_grad()
	output = net([head_img,head_pos])
	loss = GDLoss(output,direction_gt)
	print(loss.data)
	loss.backward()	
	optimizer.step()


print(output,direction_gt)
