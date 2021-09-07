import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import spatial_transforms 
from PIL import Image
import numpy as np
import glob
import random


def gen_split_mmaps(root_dir, stackSize, dir_users):
    Dataset = []
    Mmaps = []
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
                        inst_dir = os.path.join(dir1, inst+'/mmaps') # into element folder of action
                        numFrames_mmaps = len(glob.glob1(inst_dir, '*.png'))
                        numFrames_rgb = len(glob.glob1(os.path.join(dir1, inst+'/rgb'), '*.png'))
                        if numFrames_mmaps >= stackSize and numFrames_mmaps >= stackSize  >= stackSize:
                            Mmaps.append(inst_dir)
                            Dataset.append(os.path.join(dir1, inst+'/rgb'))
                            Labels.append(class_to_idx[target])
                
    return Dataset, Mmaps, Labels

class makeDatasetMmaps(Dataset):
    def __init__(self, root_dir, dir_users, numFrame, spatial_transform=None, normalize=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png'):

        self.images, self.mmaps, self.labels = gen_split_mmaps(root_dir, numFrame, dir_users)
        self.spatial_transform = spatial_transform 
        self.normalize = normalize
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.numFrame = numFrame
        self.seqLen = seqLen
        self.fmt = fmt
        
        self.spatial_transform_rgb = spatial_transforms.Compose([spatial_transforms.ToTensor(), self.normalize])
        self.spatial_transform_mmaps = transforms.Compose([transforms.Resize(7), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        vid_mmaps = self.mmaps[idx]
        label = self.labels[idx]
        inpSeq = []
        inpSeq_mmaps = []
        
        self.spatial_transform.randomize_parameters()

        for i in np.linspace(1, self.numFrame, self.numFrame, endpoint=True):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform_rgb(self.spatial_transform(img.convert('RGB'))))
            
            fl_name_mmaps = vid_mmaps + '/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt
            if not os.path.exists(fl_name_mmaps):
                fl_name_mmaps = vid_mmaps + '/' + 'map' + str(int(np.floor(i+1))).zfill(4) + self.fmt
            
            img_mmap = Image.open(fl_name_mmaps)
            inpSeq_mmaps.append(self.spatial_transform_mmaps(self.spatial_transform(img_mmap.convert('1'))))
            
           
        inpSeq = torch.stack(inpSeq, 0)
        inpSeq_mmaps = torch.stack(inpSeq_mmaps, 0)
        return inpSeq, inpSeq_mmaps, label