import json
import utils
from GazeDirectionPathway import GazeDirectionNet 
import cv2
import torch
from torchvision import transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MIT_DATA_PATH = 'example_data/MIT_annotation.txt'
MIT_ORIGINAL_PATH = 'example_data/MIT_original.jpg'
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
print(direction_gt)

net = GazeDirectionNet() 
print("######################")
output = net([head_img,head_pos])
print(output)
