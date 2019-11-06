# -*- coding: utf-8 -*-
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from config import config
from data_rgbt import TrainDataLoaderRGBT
from torch.utils.data import DataLoader
from util import util, AverageMeter, SavePlot
from got10k.datasets import ImageNetVID, GOT10k
from torchvision import datasets, transforms
from custom_transforms import Normalize, ToTensor, RandomStretch, RandomCrop, CenterCrop, RandomBlur, ColorAug
from experimentrgbt import RGBTSequence
import net

import wandb

torch.manual_seed(1234) # config.seed


parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')
parser.add_argument('--experiment_name', default='SiamRPN', metavar='DIR',help='path to weight')
parser.add_argument('--checkpoint_path', default=None, help='resume')
parser.add_argument('--modality', default=None, type=int, help='how many modalities', choices=[1, 2])

def main():

    '''parameter initialization'''
    args = parser.parse_args()
    exp_name_dir = util.experiment_name_dir(args.experiment_name)


    wandb.init(project="SiameseRPN", reinit=True)
    wandb.config.update(args)  # adds all of the arguments as config variables


    '''model on gpu'''
    model = net.TrackerSiamRPN(modality=args.modality)

    name = 'RGBT-234'

    assert name in ['VID', 'GOT-10k', 'All', 'RGBT-234']

    if name == 'GOT-10k':
        root_dir = '/home/zuern/datasets/tracking/GOT10k'
        seq_dataset = GOT10k(root_dir, subset='train')
        seq_dataset_val = GOT10k(root_dir, subset='val')

    elif name == 'VID':
        root_dir = '/home/zuern/datasets/tracking/VID/ILSVRC'
        seq_dataset = ImageNetVID(root_dir, subset=('train'))
        seq_dataset_val = ImageNetVID(root_dir, subset=('val'))

    elif name == 'RGBT-234':
        seq_dataset = RGBTSequence('/home/zuern/datasets/thermal_tracking/RGB-T234/', subset='train')
        seq_dataset_val = RGBTSequence('/home/zuern/datasets/thermal_tracking/RGB-T234/', subset='val')
    else:
        raise ValueError('Dataset not defined')


    print('seq_dataset', len(seq_dataset))
    print('seq_dataset_val', len(seq_dataset_val))


    train_z_transforms = transforms.Compose([
        RandomBlur(0.3),
        ToTensor()
    ])

    train_x_transforms = transforms.Compose([
        RandomBlur(0.3),
        ToTensor(),
    ])

    val_z_transforms = transforms.Compose([
        ToTensor()
    ])
    val_x_transforms = transforms.Compose([
        ToTensor()
    ])

    train_data  = TrainDataLoaderRGBT(seq_dataset, train_z_transforms, train_x_transforms, name)
    anchors = train_data.anchors

    train_loader = DataLoader(  dataset    = train_data,
                                batch_size = config.train_batch_size,
                                shuffle    = True,
                                num_workers= config.train_num_workers,
                                pin_memory = True)

    val_data  = TrainDataLoaderRGBT(seq_dataset_val, val_z_transforms, val_x_transforms, name)

    val_loader = DataLoader(    dataset    = val_data,
                                batch_size = config.valid_batch_size,
                                shuffle    = False,
                                num_workers= config.valid_num_workers,
                                pin_memory = True)


    '''load weights'''
    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'model' in checkpoint.keys():
            model.net.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu')['model'])
        else:
            model.net.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
        torch.cuda.empty_cache()
        print('You are loading the model.load_state_dict')


    elif config.pretrained_model:
        checkpoint = torch.load(config.pretrained_model)
        # change name and load parameters
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}
        model_dict = model.net.state_dict()
        model_dict.update(checkpoint)
        model.net.load_state_dict(model_dict)



    train_closses, train_rlosses, train_tlosses = AverageMeter(), AverageMeter(), AverageMeter()
    val_closses, val_rlosses, val_tlosses = AverageMeter(), AverageMeter(), AverageMeter()


    train_val_plot = SavePlot(exp_name_dir, 'train_val_plot')

    for epoch in range(config.epoches):

        model.net.train()

        print('Train epoch {}/{}'.format(epoch+1, config.epoches))
        train_loss = []

        with tqdm(total=config.train_epoch_size) as progbar:

            for i, dataset in enumerate(train_loader):

                closs, rloss, loss = model.step(epoch, dataset,anchors, i,  train=True)
                closs_ = closs.cpu().item()

                if np.isnan(closs_):
                   sys.exit(0)

                train_closses.update(closs.cpu().item())
                train_rlosses.update(rloss.cpu().item())
                train_tlosses.update(loss.cpu().item())

                progbar.set_postfix(closs='{:05.3f}'.format(train_closses.avg),
                                    rloss='{:05.5f}'.format(train_rlosses.avg),
                                    tloss='{:05.3f}'.format(train_tlosses.avg))

                progbar.update()
                train_loss.append(train_tlosses.avg)


        print('saving model')
        model.save(model, exp_name_dir, epoch)

        train_loss = np.mean(train_loss)

        '''val phase'''
        val_loss = []


        with tqdm(total=config.val_epoch_size) as progbar:

            print('Val epoch {}/{}'.format(epoch+1, config.epoches))

            for i, dataset in enumerate(val_loader):

                val_closs, val_rloss, val_tloss = model.step(epoch, dataset, anchors, train=False)
                closs_ = val_closs.cpu().item()

                if np.isnan(closs_):
                    sys.exit(0)

                val_closses.update(val_closs.cpu().item())
                val_rlosses.update(val_rloss.cpu().item())
                val_tlosses.update(val_tloss.cpu().item())

                progbar.set_postfix(closs='{:05.3f}'.format(val_closses.avg),
                                    rloss='{:05.5f}'.format(val_rlosses.avg),
                                    tloss='{:05.3f}'.format(val_tlosses.avg))
                progbar.update()

                val_loss.append(val_tlosses.avg)

                if i >= config.val_epoch_size - 1:
                    break


        val_loss = np.mean(val_loss)
        train_val_plot.update(train_loss, val_loss)

        print ('Train loss: {}, val loss: {}'.format(train_loss, val_loss))

        wandb.log({'Train loss': train_loss,
                   'Val loss': val_loss})


if __name__ == '__main__':
    main()
