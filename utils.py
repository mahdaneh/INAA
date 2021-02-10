import  time
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm
from tensorboardX import SummaryWriter
import sys
sys.path.append('/gel/usr/maabb14/Documents/MyCodes-Source/Augmented_CNN')
import model_building as MDL_build

def load_model (arch='resnet101', pretrained = 'imagenet'):
    if arch =='resnet101':

        model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')

        if 'scale' in pretrainedmodels.pretrained_settings[arch][pretrained]:
            scale = pretrainedmodels.pretrained_settings[arch][pretrained]['scale']
        else:
            scale = 0.875

        transform = ldb.TransformImage_nomean(model, preserve_aspect_ratio=False)
        # transform = pretrainedmodels.utils.TransformImage(model,
        #     scale=scale,
        #     preserve_aspect_ratio=False)

    elif arch=='vgg':
        model = MDL_build.vgg16_bn(num_class=10)
        chkpt = torch.load('exp_config_v3imagenette_50.pt')  # load checkpoint
        model.load_state_dict(chkpt['model'])
        transform = transforms.Compose([transforms.Resize((56,56)), transforms.ToTensor()])

    return  model, transform


def load_adapt_imagenette_imagenet(batchsize=1, transform=None):
    data = ldb.ImageFolder_imagenette(root='data/imagenette',transform=transform)
    test_loader = DataLoader(data, batchsize, shuffle=False, num_workers=0)
    return test_loader

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k*100.0 / batch_size)
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visi_input(dataloader, name):
    writer = SummaryWriter('runs')
    for i, (inputs, targets) in enumerate(tqdm.tqdm(dataloader)):
        print (torch.max(inputs))
        writer.add_image('%s_%d'%(name, i), torchvision.utils.make_grid(inputs), 0)


def eval_accuracy_imgnet(model, dataloader, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()


    # switch to train mode
    model.to(device)
    model.eval()

    end = time.time()

    for  (input, target) in ((dataloader)):


        data_time.update(time.time() - end)

        target, input = target.to(device), input.to(device)
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input.shape[0])
        top1.update(prec1, input.shape[0])
        top5.update(prec5, input.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(

          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss  ({loss.avg:.4f})\t'
          'Acc@1  ({top1.avg:.3f})\t'
          'Acc@5  ({top5.avg:.3f})'.format(
        data_time=data_time, loss=losses, top1=top1, top5=top5))


def normalize(data,std, mean, unnormalize = False):
    dtype = data.dtype
    device = data.device
    std = torch.as_tensor(std, dtype = dtype,device = device)
    std = std.view(-1,1,1,1)

    mean = torch.as_tensor(mean, dtype = dtype,device = device)
    mean = mean.view(-1,1,1,1)

    if unnormalize:
        data = (data *std)+mean
    else:
        data = (data - mean) / std


    return data

import pretrainedmodels
import load_data as ldb

def load_model (arch='resnet101', pretrained = 'imagenet'):

    model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')

    if 'scale' in pretrainedmodels.pretrained_settings[arch][pretrained]:
        scale = pretrainedmodels.pretrained_settings[arch][pretrained]['scale']
    else:
        scale = 0.875

    return  model
