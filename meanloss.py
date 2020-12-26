import json
import numpy as np
with open('batchsize_5_lr_0.0001_5_epoch_pretrained/loss.json','r') as f:
    lst = json.load(f)
print(lst[0][0],lst[1][0],lst[2][0])

