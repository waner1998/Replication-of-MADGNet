import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from MADGNet.IS2D_models.mfmsnet import MFMSNet
import time

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument('--gpu_choose', dest='gpu_choose', type=int, default=0,
                    help='the number of gpu')

parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--num_workers", type=int, default=0,
                    help="num workers of dataloader")
parser.add_argument("--lr", type=float, default=1E-4, help="learning rate")
parser.add_argument("--batch_size", default=16, type=int)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def mse_loss(pred, mask):
    loss = F.mse_loss(pred, mask)
    return loss

def bce_loss(pred, mask):
    loss = F.binary_cross_entropy_with_logits(pred, mask)
    return loss


def main(args):    
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    device = torch.device("cuda:{}".format(args.gpu_choose) if torch.cuda.is_available() else "cpu")
    model = MFMSNet()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-6)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):
        time1 = time.time()
        for i, batch in enumerate(dataloader):

            images = batch['image']
            gts = batch['label']
            images = images.to(device)
            gts = gts.to(device)
            optim.zero_grad()

            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5[0], gts) + mse_loss(lateral_map_5[1], gts) + bce_loss(lateral_map_5[2],
                                                                                                       gts)
            loss4 = structure_loss(lateral_map_4[0], gts) + mse_loss(lateral_map_4[1], gts) + bce_loss(lateral_map_4[2],
                                                                                                       gts)
            loss3 = structure_loss(lateral_map_3[0], gts) + mse_loss(lateral_map_3[1], gts) + bce_loss(lateral_map_3[2],
                                                                                                       gts)
            loss2 = structure_loss(lateral_map_2[0], gts) + mse_loss(lateral_map_2[1], gts) + bce_loss(lateral_map_2[2],
                                                                                                       gts)
            loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
            loss.backward()
            optim.step()
            '''
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
            '''
        time2 = time.time()
        print("epoch:{}/{}: loss:{}, time spent: {:.2f} s".format(epoch + 1, args.epoch, loss.item(), time2 - time1))
        scheduler.step()
        if epoch + 1 == args.epoch:
            torch.save(model.state_dict(), args.save_path + '/MADGNet_{}.pth'.format(epoch+1))


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main(args)