import os
import torch
import pickle
from model import AU, DSAU
import torch.backends.cudnn as cudnn
import Dataset as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
import shutil
from loss import TotalLoss

def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    if args.model_type == 'DSAU':
        model = DSAU.DS_attention_unet()
    else:
        model = AU.attention_unet()

    args.savedir = args.savedir + '/' + args.model_type + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    else:
        shutil.rmtree(args.savedir)
        os.mkdir(args.savedir)

    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.AdamW(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #             .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))


    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        model.train()
        train(args, trainLoader, model, criteria, optimizer, epoch)
        model.eval()
        # validation
        da_segment_results=val(valLoader, model)
        msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'.format( \
                        da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2])
        print(msg)
        torch.save(model.state_dict(), model_file_name)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='DSAU', help='DSAU: Depthwise Separable Attention Unet; AU: Attention Unet')
    parser.add_argument('--num_workers', type=int, default=16, help='No. of parallel threads')
    parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate')
    parser.add_argument('--savedir', default='workdirs/', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')

    train_net(parser.parse_args())

