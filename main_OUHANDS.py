import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.model import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import TSNE
import datetime
from dataloader.dataset_OUHANDS import train_data_loader, test_data_loader
from collections import Counter
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--dataset', type=str, default='cityscapes', help='pascal or cityscapes')
parser.add_argument('--no_of_classes', type=int, default=None, help='no of classes')
parser.add_argument('--subject', type=int, default=None, help='subject choosen as test set')
parser.add_argument('--channels', type=int, default=1, help='no of classes')
parser.add_argument('--train_split', type=str, default=None, help='train split')
parser.add_argument('--val_split', type=str, default=None, help='val split')
parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--data_set', type=int, default=1)
parser.add_argument('--gpu-ids', default='0,1,2,3', type=str)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# print("cuda:",args.cuda)
if args.cuda:
    try:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        # print("IDS:",args.gpu_ids)
    except ValueError:
        raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

if args.sync_bn is None:
    if args.cuda and len(args.gpu_ids) > 1:
        args.sync_bn = True
    else:
        args.sync_bn = False
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
log_txt_path = './log/' +'subject_'+str(args.subject)+ time_str + 'log.txt'
log_curve_path = './log/' +'subject_'+str(args.subject)+ time_str + 'log.png'
checkpoint_path = './checkpoint/' +'subject_'+str(args.subject)+ time_str + 'model.pth'
best_checkpoint_path = './checkpoint/' +'subject_'+str(args.subject)+ time_str + 'model_best.pth'



def main():
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))

    # create model and load pre_trained parameters
    model = GenerateModel()
   
    if args.cuda and len(args.gpu_ids)>1:
        model = nn.DataParallel(model).cuda()
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index = -1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    train_data, val_data = train_data_loader()
    

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    test_dataset = test_data_loader()
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_los, train_conf_matrix = train(train_loader, model, criterion, optimizer, epoch, args)
        print("train_acc:",train_acc)
        # evaluate on validation set
        val_acc, val_los, val_conf_matrix = validate(val_loader, model,'Val', criterion, args)
        print("val_acc:",val_acc)
        scheduler.step()

        print("Train Confusion Matrix:")
        print(train_conf_matrix)
        print("Validation Confusion Matrix:")
        print(val_conf_matrix)

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')
     # After training and validation, use the model for testing
    model_best = model
    model_best.eval()  # Set the model to evaluation mode

    # Load the best model checkpoint for testing
    checkpoint_best = torch.load(best_checkpoint_path)  # Load the best model checkpoint
    model_best.load_state_dict(checkpoint_best['state_dict'])  # Load model state dict

    # Prepare the testing data loader

    # Evaluate the model on the testing set
    test_acc_best, test_loss_best, test_conf_matrix_best = validate(test_loader, model_best,'Test_best', criterion, args)

    print("Test best Confusion Matrix:")
    print("test_conf_matrix_best",test_conf_matrix_best)

    # Print and save test results
    print('Test accuracy best model: {:.3f}'.format(test_acc_best))
    with open(log_txt_path, 'a') as f:
        f.write('Test best accuracy: {:.3f}\n'.format(test_acc_best))

    ###########################################################
    model.eval()  # Set the model to evaluation mode

    # Load the best model checkpoint for testing
    checkpoint = torch.load(checkpoint_path)  # Load the best model checkpoint
    model.load_state_dict(checkpoint['state_dict'])  # Load model state dict

    # Prepare the testing data loader

    # Evaluate the model on the testing set
    test_acc, test_loss, test_conf_matrix = validate(test_loader, model,'Test', criterion, args)

    # print("Validation Confusion Matrix:")
    print("test_conf_matrix",test_conf_matrix)

    # Print and save test results
    print('Test accuracy  model: {:.3f}'.format(test_acc))
    with open(log_txt_path, 'a') as f:
        f.write('Test accuracy: {:.3f}\n'.format(test_acc))

    


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    tbar = tqdm(train_loader)
    args.no_of_classes=10
    all_preds=[]
    all_targets=[]   
    for i, sample in enumerate(tbar):
        cur_iter = epoch * len(train_loader) + i
        inputs = sample['image'].cuda()
        target = sample['label'].cuda()

        outputs,_ = model(inputs)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        loss = criterion(outputs, target)
 

        k = 5 if args.no_of_classes>=5 else args.no_of_classes
        acc1, acc5 = accuracy(outputs,target,topk=(1,k))

        n = inputs.size(0)

        top1.update(acc1.item(), n)
        losses.update(loss.item(), n)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tbar.set_description('Train loss: %.3f' % losses.avg)
    confusion_mat = confusion_matrix(all_targets, all_preds)        
    return top1.avg, losses.avg, confusion_mat


def validate(val_loader, model,ar, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    args.no_of_classes=10
    tbar = tqdm(val_loader, desc='\r')

    all_preds=[]
    all_targets=[]
    with torch.no_grad():
        scatter_all_x = []
        scatter_all_y = []
      
        for i, sample in enumerate(tbar):
            inputs = sample['image'].cuda()
            target = sample['label'].cuda()
            outputs, scatter = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            loss = criterion(outputs, target)
            ##TSNE
            scatter_all_x.extend(scatter.cpu().numpy())
            scatter_all_y.extend(target.cpu().numpy())

            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            k = 5 if args.no_of_classes>=5 else args.no_of_classes
            acc1, acc5 = accuracy(outputs,target,topk=(1,k))
            n = inputs.size(0)
            top1.update(acc1.item(), n)
            losses.update(loss.item(), n)
            
            tbar.set_description('Valid loss: %.3f' % losses.avg)
        confusion_mat = confusion_matrix(all_targets, all_preds)
        tSNE_x = np.array(scatter_all_x)
        tSNE_y = np.array(scatter_all_y)
        TSNE.tsne(tSNE_x, tSNE_y, str(args.data_set), ar, 1)  
    return top1.avg, losses.avg, confusion_mat


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
