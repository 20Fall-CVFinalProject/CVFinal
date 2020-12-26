import resnet
import utils
import json
import torch
import torch.nn as nn

# def ThreeFC(hx,hy): #(hx,hy) is the head position which can be received from
# with open('example_data/SVIP_annotationn.json','r') as ann:
#     data_svip = json.load(ann)
# print("SVIP:", data_svip)

# with open('example_data/MIT_annotation.txt','r') as ann:
#     data_mit = ann.read()
# print("MIT:",data_mit)

class GazeDirectionNet(nn.Module): 
    #reference:gazenet.py, https://github.com/svip-lab/GazeFollowing/blob/master/code/gazenet.py
    def __init__(self):
        super(GazeDirectionNet,self).__init__()
        # head_feature extraction: input is headimage which is 3-dim,the output is 512-dim
        self.head_feature_net = resnet.resnet50(True) # note to add pretrained!!!!  #for resnet50, the out_feature is 2048-dim
        out_feature = 2048
        self.head_feature_process = nn.Sequential(nn.Linear(out_feature,512),nn.ReLU(inplace = True))
        
        # head_position_feature extraction: input is coordinate(hx,hy) which is 2-dim, output is 256-dim
        self.head_pos_net = nn.Sequential(nn.Linear(2,256),nn.ReLU(inplace = True),
                                          nn.Linear(256,256),nn.ReLU(inplace = True),
                                          nn.Linear(256,256),nn.ReLU(inplace = True))#different from the reference
        
        #concatenate head features with head position features
        self.concatenate_net = nn.Sequential(nn.Linear(768,256),nn.ReLU(inplace = True),
                                            nn.Linear(256,2))
    
    def forward(self,x):
        head_img, head_pos = x #TBD
        
        head_feature = self.head_feature_process(self.head_feature_net(head_img))
        
        head_pos_feature =self.head_pos_net(head_pos)
        direction = self.concatenate_net(torch.cat((head_feature,head_pos_feature),1))
        norm = torch.norm(direction, 2, dim = 1)
        normalized_direction = direction / norm.view([-1, 1])
        return normalized_direction