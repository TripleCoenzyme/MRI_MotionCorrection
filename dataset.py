import os
import torch
from scipy import io
from torchvision import transforms
from torch import nn
import numpy as np

np.random.seed(17)
#7.25定稿版本，用于比赛测试

class Dataset(torch.utils.data.Dataset):
    '''
    给定三种取数据mode：'train'/'test'/'validation'
    '''
    def __init__(self, path, transform=transforms.Pad((6,0,7,0)), mode='train', val_path = None):
        self.img_txt = path+'/img.txt'
        self.label_txt = path+'/label.txt'
        self.mm_txt = path+'/mm.txt'
        self.max = []
        self.transform = transform
        split = 0.95
        with open(self.mm_txt,'r') as f:
            self.nii_list = [i.split('\t')[0] for i in f.readlines()]
            np.random.shuffle(self.nii_list)
        self.mode = mode
        if self.mode == 'train':
            with open(self.img_txt,'r') as f:
                self.img_log = [path+i.lstrip('./').rstrip() for i in f.readlines() if i.split('\t')[0] in self.nii_list[:int(len(self.nii_list)*split)]]

        elif self.mode == 'val':
            with open(self.img_txt,'r') as f:
                #self.img_log = [path+i.lstrip('./').rstrip() for i in f.readlines() if i.split('\t')[0] in self.nii_list[int(len(self.nii_list)*split):]]
                self.img_log = [path+i.lstrip('./').rstrip() for i in f.readlines() if i.split('\t')[0] in [self.nii_list[-1]]]
        
        '''elif self.mode == 'test':
            if not(os.path.isfile(path+'test.txt')):
                utils.get_img_list(path, 'test')
            with open(path+'test.txt','r') as f_test:
                log = f_test.readlines()'''
        self.label_log = []
        self.ma = []
        with open(self.mm_txt,'r') as f:
            tmp = f.readlines()
        for i in self.img_log:
            for k in tmp:
                if k.split('\t')[0][2:] in i.split('\t')[0]:
                    self.ma.append(float(k.split('\t')[1]))
                    break
            j = i.split('_')[:8]
            j.extend(['label',i.split('_')[8]+'.mat'])
            self.label_log.append('_'.join(j))

    def __getitem__(self,index):
        img_path = "/".join(self.img_log[index].split('\t'))
        label_path = "/".join(self.label_log[index].split('\t'))
        img = torch.tensor(io.loadmat(img_path)["post_slice"]/self.ma[index]).to(torch.float32)
        label = torch.tensor(io.loadmat(label_path)["slice"]/self.ma[index]).to(torch.float32)
        img = torch.where(img>1e-8, img, torch.zeros_like(img))
        label = torch.where(label>1e-8, label, torch.zeros_like(label))
        return [self.transform(img.unsqueeze_(0)),self.transform(label.unsqueeze_(0))]

    def __len__(self):
        return len(self.img_log)
