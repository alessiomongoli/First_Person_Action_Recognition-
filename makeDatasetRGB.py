import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, stackSize, dir_users):
    Dataset = []
    Labels = []
    
    classes = []
    
    for user in ['S1','S2','S3','S4']:
        user_dir = os.path.join(root_dir, user)
        classes.extend(dir for dir in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, dir)))
    
    classes = list(set(classes))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
        
    for dir_user in dir_users:

        dir = os.path.join(root_dir, dir_user)

        for target in sorted(os.listdir(dir)): # into folder user
            dir1 = os.path.join(dir, target) 
            if os.path.isdir(dir1):
                insts = sorted(os.listdir(dir1)) # into single action folder
                if insts != []:
                    for inst in insts:
                        inst_dir = os.path.join(dir1, inst+'/rgb') # into element folder of action
                        numFrames = len(glob.glob1(inst_dir, '*.png'))
                        if numFrames >= stackSize:
                            Dataset.append(inst_dir)
                            Labels.append(class_to_idx[target])
                
    return Dataset, Labels

class makeDataset(Dataset):
    def __init__(self, root_dir, dir_users, numFrame, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png'):

        self.images, self.labels = gen_split(root_dir, numFrame, dir_users)
        self.spatial_transform = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.numFrame = numFrame
        self.seqLen = seqLen
        self.fmt = fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, self.numFrame, self.numFrame, endpoint=True):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
