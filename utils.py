import  time
import torch
from torch import nn
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

def eval_accuracy_imgnet(model, dataloader, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()

    # switch to train mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(dataloader):

        # measure data loading time
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
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        data_time=data_time, loss=losses, top1=top1, top5=top5))

