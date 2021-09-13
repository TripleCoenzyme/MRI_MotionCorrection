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

###########
#可调整的训练超参数
batch_size = 16
val_batch_size = 16
lr = 1e-4
start_epoch = 40
stop_epoch = 60
###########

###########
#可调整的路径参数
title = 'mo_l1_1213'
path = 'C:/Users/FengJie/Desktop/motion_correct/'
data_path = 'Y:/lzh_znso4/interventional/silce/'
Model_path = path+'log/checkpoints/'+title+'/40.pth'
###########

###########
#可调整的训练相关处理
pretrain = True
multi_GPU = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_step = 100 #决定多少次保存一次可视化结果
###########

###########
#无需调整的路径参数
log_path = path+'log/'
checkpoints_path = path+'log/checkpoints/'+title+'/'
tensorboard_path = path+'log/tensorboard/'+title+'/'
visualize_path = path+'log/visualize/'+title+'/'
###########


if __name__ == '__main__':
    path_checker(log_path)
    path_checker(checkpoints_path)
    path_checker(tensorboard_path)
    path_checker(visualize_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    Writer = SummaryWriter(tensorboard_path)

    train_set = Dataset(path=data_path, mode='train')
    val_set = Dataset(path=data_path, mode='val')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False)

    model = Resnet_Unet().to(device)
    if pretrain:
        model.load_state_dict(torch.load(Model_path))
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam([{'params':model.parameters(), 'initial_lr': lr}],lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=start_epoch-1)

    for epoch in range(start_epoch, stop_epoch):
        batch_sum = len(train_loader)
        
        # 训练部分
        model.train()
        for index, (input, label) in enumerate(train_loader):
            
            input = input.to(device)
            label = label.to(device)

            for param in model.parameters():
                param.grad = None
            output = model(input)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            Writer.add_scalar('scalar/loss', loss, epoch*batch_sum+index)
            if index % save_step == 0:
                input_img = make_grid(input.cpu()[0, 0, :, :], padding=2, normalize=True).detach()*255
                label_img = make_grid(label.cpu()[0, 0, :, :], padding=2, normalize=True).detach()*255
                output_img = make_grid(output.cpu()[0, 0, :, :], padding=2, normalize=True).detach()*255

                Writer.add_image('image/input', input_img.to(torch.uint8), epoch*batch_sum+index)
                Writer.add_image('image/output', output_img.to(torch.uint8), epoch*batch_sum+index)
                Writer.add_image('image/label', label_img.to(torch.uint8), epoch*batch_sum+index)
                skimage.io.imsave(visualize_path + str(epoch+1) + '_' + str(index) + '.jpg', torch.cat([input_img,label_img,output_img],1).to(torch.uint8).cpu().numpy().transpose((1, 2, 0)))
            sys.stdout.write(
                "\r[Train] [Epoch {}/{}] [Batch {}/{}] [loss:{:.8f}] [learning rate:{:.8e}]".format(epoch + 1, stop_epoch,
                                                                                                index + 1, batch_sum,
                                                                                                loss.item(),
                                                                                                optimizer.param_groups[0][
                                                                                                    'lr']))
            sys.stdout.flush()
        print('\n')
        torch.save(model.state_dict(), checkpoints_path + '{}.pth'.format(epoch + 1))


        model.eval()
        with torch.no_grad():
            loss = []
            for index, (input, label) in enumerate(val_loader):
            
                input = input.to(device)
                label = label.to(device)
                output = model(input)
                loss.append(criterion(output, label).item())


                stdout = '\r[Val] [Epoch {}/{}] [Batch {}/{}] [learning rate:{}] [loss:{:.8f}]\n'.format(epoch + 1, stop_epoch,
                                                                                                index + 1, len(val_loader),
                                                                                                optimizer.param_groups[0]['lr'], 
                                                                                                loss[index]
                                                                                                )
                sys.stdout.write(stdout)
                
                '''with open(checkpoints_path+'log.txt','a') as f:
                    f.write(stdout[:-1])
                sys.stdout.flush()'''
            with open(checkpoints_path+'log.txt','a') as f:
                f.write('[Val] [Epoch {}/{}] [loss:{:.8f}]\n'.format(epoch + 1, stop_epoch, mean(loss)))
        #scheduler.step()