import os
import sys
import time
from numpy.core.fromnumeric import mean

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import numpy as np
from dataset import Dataset
from scipy import io
import skimage.io
from skimage import metrics
from model import Resnet_Unet

def path_checker(path):
    """
    检查目录是否存在，不存在，则创建
    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print(path+'不存在，已创建...')
    else:
        print(path+'已存在')


batch_size = 16
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
path = 'D:/motion_correct/'
data_path = 'Y:/lzh_znso4/interventional/silce/'
Model_path = path+'log/checkpoints/mo_patchGAN_1215_test/G_15.pth'
visualize_path = 'D:/motion_correct/log/visualize/test_15_GAN/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val_set = Dataset(path=data_path, mode='val')
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
model = Resnet_Unet().to(device)
model.load_state_dict(torch.load(Model_path))

path_checker(visualize_path)

model.eval()
psnr = []
ssim = []
with torch.no_grad():
    for index, (input, label) in enumerate(val_loader):
    
        input = input.to(device)
        label = label.to(device)
        #output = model(input)
        #loss = criterion(output, label)
        for i in range(input.shape[0]):
            psnr.append(metrics.peak_signal_noise_ratio(label[i,0,:,:].squeeze().cpu().numpy(), input[i,0,:,:].squeeze().cpu().numpy(), data_range=1))
            ssim.append(metrics.structural_similarity(label[i,0,:,:].squeeze().cpu().numpy(), input[i,0,:,:].squeeze().cpu().numpy(),data_range=1.0))

            #skimage.io.imsave(visualize_path + str(index*batch_size+i+1) + '.png', (torch.cat([input[i,0,:,:].squeeze(),label[i,0,:,:].squeeze(),output[i,0,:,:].squeeze()],1)*255).to(torch.uint8).cpu().numpy())
            #skimage.io.imsave(visualize_path + str(index*batch_size+i+1) + 'diff.png', (torch.cat([(input[i,0,:,:].squeeze()-label[i,0,:,:].squeeze()+0.5),(output[i,0,:,:].squeeze()-label[i,0,:,:].squeeze()+0.5)],1)*255).to(torch.uint8).cpu().numpy())
        sys.stdout.write('\r[Val] [Batch {}/{}] [psnr:{:.8f}] [ssim:{:.8f}]'.format(index + 1, len(val_loader),
                                                                                        np.mean(psnr), np.mean(ssim)
                                                                                        ))
        sys.stdout.flush()
    print('\n')
        