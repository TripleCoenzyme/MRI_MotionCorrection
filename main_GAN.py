import os
import sys
import time

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
from model import Resnet_Unet, Discriminator

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
start_epoch = 0
stop_epoch = 20
###########

###########
#可调整的路径参数
title = 'mo_patchGAN_1219_test_origin'
path = 'D:/motion_correct/'
data_path = 'Y:/lzh_znso4/interventional/silce/'
Model_path = path+'log/checkpoints/mo_patchGAN_1218_test/G_5.pth'
D_path = path+'log/checkpoints/mo_patchGAN_1218_test/D_5.pth'
###########

###########
#可调整的训练相关处理
pretrain = False
multi_GPU = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_step = 300 #决定多少次保存一次可视化结果
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
    D = Discriminator().to(device)
    if pretrain:
        model.load_state_dict(torch.load(Model_path))
        D.load_state_dict(torch.load(D_path))

    criterion_G = nn.L1Loss().to(device)
    criterion_D = nn.BCELoss().to(device)
    optimizer_G = torch.optim.Adam([{'params':model.parameters(), 'initial_lr': lr}],lr=lr)
    optimizer_D = torch.optim.Adam([{'params':D.parameters(), 'initial_lr': lr}],lr=lr)
    #scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.1, last_epoch=start_epoch-1)
    #scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.1, last_epoch=start_epoch-1)

    for epoch in range(start_epoch, stop_epoch):
        batch_sum = len(train_loader)

        #训练D
        for index, (input, label) in enumerate(train_loader):
            input = input.to(device)
            label = label.to(device)

            D.train()
            model.eval()

            for param in D.parameters():
                param.grad = None

            output = D(label)
            errD_real = criterion_D(output, torch.ones_like(output))
            errD_real.backward()

            fake = model(input)
            output = D(fake.detach())
            errD_fake = criterion_D(output, torch.zeros_like(output))
            errD_fake.backward()

            optimizer_D.step()
            D_loss = (errD_real + errD_fake)
            Writer.add_scalar('scalar/D_loss', D_loss, epoch*batch_sum+index)
            
            
            D.eval()
            model.train()

            for param in model.parameters():
                param.grad = None

            output = D(fake)
            l1 = criterion_G(fake, label)
            bc = criterion_D(output, torch.ones_like(output))
            errG = 0.001*bc+l1#####
            errG.backward()

            optimizer_G.step()
            Writer.add_scalar('scalar/G_loss', errG, epoch*batch_sum+index)

            if index % save_step == 0:
                input_img = make_grid(input.cpu()[0, 0, :, :], padding=2, normalize=True).detach()*255
                label_img = make_grid(label.cpu()[0, 0, :, :], padding=2, normalize=True).detach()*255
                output_img = make_grid(fake.cpu()[0, 0, :, :], padding=2, normalize=True).detach()*255

                Writer.add_image('image/input', input_img.to(torch.uint8), epoch*batch_sum+index)
                Writer.add_image('image/output', output_img.to(torch.uint8), epoch*batch_sum+index)
                Writer.add_image('image/label', label_img.to(torch.uint8), epoch*batch_sum+index)
                skimage.io.imsave(visualize_path + str(epoch+1) + '_' + str(index) + '.png', torch.cat([input_img,label_img,output_img],2).to(torch.uint8).cpu().numpy().transpose((1, 2, 0)))
                torch.save(model.state_dict(), checkpoints_path + 'G_{}_{}.pth'.format(epoch + 1, index))
            sys.stdout.write(
                "\r[Train] [Epoch {}/{}] [Batch {}/{}] [D_loss:{:.8f}] [BC_loss:{:.8f}] [L1_loss:{:.8f}] [learning rate:{:.8e}]".format(epoch + 1, stop_epoch,
                                                                                                index + 1, batch_sum,
                                                                                                D_loss,
                                                                                                bc,
                                                                                                l1,
                                                                                                optimizer_D.param_groups[0][
                                                                                                    'lr']))
            sys.stdout.flush()
        print('\n')
        torch.save(model.state_dict(), checkpoints_path + 'G_{}.pth'.format(epoch + 1))
        torch.save(D.state_dict(), checkpoints_path + 'D_{}.pth'.format(epoch + 1))
        #scheduler.step()


        model.eval()
        psnr = []
        ssim = []
        with torch.no_grad():
            for index, (input, label) in enumerate(val_loader):
            
                input = input.to(device)
                label = label.to(device)
                output = model(input)
                #loss = criterion(output, label)
                for i in range(output.shape[0]):
                    psnr.append(metrics.peak_signal_noise_ratio(label[i,0,:,:].squeeze().cpu().numpy(), output[i,0,:,:].squeeze().cpu().numpy(), data_range=1))
                    ssim.append(metrics.structural_similarity(label[i,0,:,:].squeeze().cpu().numpy(), output[i,0,:,:].squeeze().cpu().numpy(),data_range=1.0))

                sys.stdout.write('\r[Val] [Epoch {}/{}] [Batch {}/{}] [psnr:{:.8f}] [ssim:{:.8f}]'.format(epoch + 1, stop_epoch,
                                                                                                index + 1, len(val_loader),
                                                                                                np.mean(psnr), np.mean(ssim)
                                                                                                ))
                sys.stdout.flush()
            print('\n')
                
                
            with open(checkpoints_path+'log.txt','a') as f:
                f.write('[Val] [Epoch {}/{}] [psnr:{:.8f}] [ssim:{:.8f}]\n'.format(epoch + 1, stop_epoch,
                                                                                np.mean(psnr), np.mean(ssim)
                                                                                ))