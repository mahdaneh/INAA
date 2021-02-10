import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import load_data as ldb
import torch

from torch.utils.data import DataLoader
import  torch.optim as optim

from collections import defaultdict
import os

import argparse
from torchvision import transforms
import torchvision
import sys
sys.path.append('/gel/usr/maabb14/Documents/MyCodes-Source/Augmented_CNN')
import model_building as MDL_build
import util as base_util
print ('util.py path: ',base_util.__file__)

import training_op as op
transfer = True
train=False
def main():
    parser = argparse.ArgumentParser()
    earlystopping = base_util.EarlyStopping()
    parser.add_argument('--config-file', type=str, default=None)
    # --pretrain-weight exp_config_v3cifar10_50000_50.pt
    parser.add_argument('--pretrain-weight', type=str, default=None)
    args = parser.parse_args()
    print(args)

    local_op = op.Local_OP(args.config_file)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    transform_tr = transforms.Compose([transforms.Resize((local_op.image_size,local_op.image_size)), transforms.RandomHorizontalFlip(), torchvision.transforms.ColorJitter(),
        transforms.ToTensor(),transforms.Normalize(mean, std)])

    transform_val = transforms.Compose([transforms.Resize((local_op.image_size,local_op.image_size)),
                                       transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    data_tr = torchvision.datasets.ImageFolder(root='data/imagenette2-320/train/', transform=transform_tr)


    train_loader = DataLoader(data_tr, local_op.batch_size, shuffle=True, num_workers=0)

    data_valid = torchvision.datasets.ImageFolder(root='data/imagenette2-320/val/', transform=transform_val)
    valid_loader = DataLoader(data_valid, local_op.batch_size, shuffle=False, num_workers=0)
    print ('training set size %d, validation set size %d'%(len(data_tr),len(data_valid)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MDL_build.vgg16_bn(num_class=10)
    if train:
        optimizer = optim.SGD(model.parameters(), lr=local_op.lr, momentum=local_op.momentum,
                              weight_decay=local_op.weight_decay)
        milestones = [20, 40]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        if transfer:
            print('**** LOADING from Pre-trained CNN')

            chkpt = torch.load(args.pretrain_weight)  # load checkpoint
            model.load_state_dict(chkpt)

        train_loss, validation_loss  = {'loss':[],'acc':[]},{'loss':[],'acc':[]}
        model.to(device)
        for epoch in range( local_op.epochs + 1):


            local_op.train(model, device, train_loader, optimizer, epoch)
            lr_scheduler.step()

            tr_acc, tr_conf, tr_loss = local_op.test(model, device, train_loader, 'train')
            train_loss['loss'].append(tr_loss)
            train_loss['acc'].append(tr_acc)



            chkpt = {'epoch': epoch,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(chkpt, os.path.join( local_op.conf_fname + local_op.dataset_name + '_' + str(local_op.epochs) + '.pt'))

            val_acc, val_conf, val_loss = local_op.test(model, device, valid_loader, 'validation')
            validation_loss['loss'].append(val_loss)
            validation_loss['acc'].append(val_acc)
            earlystopping(val_loss, model)

            if earlystopping.early_stop:
                print("Early stopping")
                break

            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)

            plt.plot(range(len(train_loss['loss'])), train_loss['loss'], label='train')
            plt.plot(range(len(train_loss['loss'])), validation_loss['loss'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(range(len(train_loss['loss'])), train_loss['acc'], label='train')
            plt.plot(range(len(train_loss['loss'])), validation_loss['acc'], label='val')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(local_op.dir, local_op.conf_fname + local_op.dataset_name + '_' + str(local_op.tr_smpl)))
            plt.close()
    else:
        data_val = torchvision.datasets.ImageFolder(root='data/imagenette', transform=transform_val)

        print (data_val.class_to_idx)
        chkpt = torch.load(args.pretrain_weight)  # load checkpoint
        model.load_state_dict(chkpt['model'])
        model.to(device)

        data_val = ldb.load_local_data(image_path='Clean_image_FGS.pt', label_path='clean_imagenette_label_FGS',
                                           transform=transform_val, convert_2_10cls=True)

        valid_loader = DataLoader(data_val, local_op.batch_size, shuffle=False, num_workers=0)
        local_op.test( model, device, valid_loader, 'test')

if __name__=='__main__':
    main()
