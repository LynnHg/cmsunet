import time
import os
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from datasets import myops2020
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from utils.loss import *
from utils.metrics import diceCoeffv2
from utils import misc
from utils.pytorchtools import EarlyStopping
from utils.LRScheduler import PolyLR
from networks.cmsunt import CMSUNet

crop_size = 256
batch_size = 6
n_epoch = 300
lr_scheduler_eps = 1e-3
lr_scheduler_patience = 10
early_stop_patience = 20
initial_lr = 1e-4
threshold_lr = 1e-6
weight_decay = 1e-5
optimizer_type = 'adam'  # adam, sgd
scheduler_type = 'no'  # ReduceLR, StepLR, poly
label_smoothing = 0.01
aux_loss = False

root_path = '/'
fold = 1  # 1, 2, 3, 4, 5
loss_name = 'dual_'
model_name = 'CMSUNet'
data_modal = ''
additional_instructions = ''

writer = SummaryWriter(os.path.join(root_path, 'log/MyoPS2020/train',
                                    model_name + data_modal + '_{}fold_{}'.format(fold, additional_instructions) + str(
                                        int(time.time()))))
val_writer = SummaryWriter(os.path.join(
    os.path.join(root_path, 'log/MyoPS2020/val', model_name + data_modal) + '_{}fold_{}'.format(fold,
                                                                                                additional_instructions) + str(
        int(time.time()))))

train_path = os.path.join(root_path, 'media/LIBRARY/Datasets/MyoPS2020/Augdata', 'train{}'.format(fold), 'npy')
val_path = os.path.join(root_path, 'media/LIBRARY/Datasets/MyoPS2020/npy')


def main():
    # 定义网络
    net = CMSUNet(num_classes=myops2020.num_classes, depth=3).cuda()

    # 数据预处理，加载
    center_crop = joint_transforms.CenterCrop(crop_size)
    input_transform = extended_transforms.NpyToTensor()
    target_transform = extended_transforms.MaskToTensor()

    train_set = myops2020.MyoPS2020(train_path, 'train', fold, joint_transform=None, roi_crop=None,
                                    center_crop=center_crop,
                                    transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = myops2020.MyoPS2020(val_path, 'val', fold,
                                  joint_transform=None, transform=input_transform, roi_crop=None,
                                  center_crop=center_crop,
                                  target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 定义损失函数
    if loss_name == 'dice_':
        criterion = SoftDiceLoss(myops2020.num_classes).cuda()
    elif loss_name == 'bce_':
        criterion = nn.BCELoss().cuda()
    elif loss_name == 'dual_':
        criterion = BCE_Dice_Loss(myops2020.num_classes).cuda()
    else:
        pass

    # 定义早停机制
    early_stopping = EarlyStopping(early_stop_patience, verbose=True, delta=lr_scheduler_eps,
                                   path=os.path.join(root_path, 'checkpoint',
                                                     '{}.pth'.format(model_name)))

    # 定义优化器
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    # 定义学习率衰减策略
    if scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    elif scheduler_type == 'ReduceLR':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif scheduler_type == 'poly':
        scheduler = PolyLR(optimizer, max_iter=n_epoch, power=0.9)
    else:
        scheduler = None

    train(train_loader, val_loader, net, criterion, optimizer, scheduler, None, early_stopping, n_epoch, 0)


def train(train_loader, val_loader, net, criterion, optimizer, scheduler, warm_scheduler, early_stopping, num_epoches,
          iters):
    for epoch in range(1, num_epoches + 1):
        st = time.time()
        train_class_dices = np.array([0] * (myops2020.num_classes - 1), dtype=np.float)
        val_class_dices = np.array([0] * (myops2020.num_classes - 1), dtype=np.float)
        train_losses = []
        val_losses = []

        # 训练模型
        net.train()
        for batch, ((inputs, mask), file_name, _) in enumerate(train_loader, 1):
            X1 = inputs[0].cuda()
            X2 = inputs[1].cuda()
            X3 = inputs[2].cuda()
            y = mask.cuda()
            optimizer.zero_grad()
            output = net(X1, X2, X3)
            output = torch.sigmoid(output)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            iters += 1
            train_losses.append(loss.item())

            class_dice = []
            for i in range(1, myops2020.num_classes):
                cur_dice = diceCoeffv2(output[:, i:i + 1, :], y[:, i:i + 1, :], activation=None).cpu().item()
                class_dice.append(cur_dice)

            mean_dice = sum(class_dice) / len(class_dice)
            train_class_dices += np.array(class_dice)
            string_print = 'epoch: {} - iters: {} - loss: {:.4} - mean: {:.4} - lvm: {:.4}- lv: {:.4} - rv: {:.4} - lvme: {:.4} - lvms: {:.4} - time: {:.2}' \
                .format(epoch, iters, loss.data.cpu(), mean_dice, class_dice[0], class_dice[1], class_dice[2],
                        class_dice[3], class_dice[4], time.time() - st)
            misc.log(string_print)
            st = time.time()

        train_loss = np.average(train_losses)
        train_class_dices = train_class_dices / batch
        train_mean_dice = train_class_dices.sum() / train_class_dices.size

        writer.add_scalar('main_loss', train_loss, epoch)
        writer.add_scalar('main_dice', train_mean_dice, epoch)

        print(
            'epoch {}/{} - train_loss: {:.4} - train_mean_dice: {:.4} - dice_lvm: {:.4} - dice_lv: {:.4} - dice_rv: {:.4} - dice_lvme: {:.4} - dice_lvms: {:.4}'.format(
                epoch, num_epoches, train_loss, train_mean_dice, train_class_dices[0], train_class_dices[1],
                train_class_dices[2], train_class_dices[3], train_class_dices[4]))

        # 验证模型
        net.eval()
        for val_batch, ((inputs, mask), file_name, _) in enumerate(val_loader, 1):
            val_X1 = inputs[0].cuda()
            val_X2 = inputs[1].cuda()
            val_X3 = inputs[2].cuda()
            val_y = mask.cuda()
            pred = net(val_X1, val_X2, val_X3)
            pred = torch.sigmoid(pred)
            val_loss = criterion(pred, val_y)

            val_losses.append(val_loss.item())
            pred = pred.cpu().detach()
            pred[pred < 0.5] = 0
            pred[pred > 0.51] = 1.0
            val_class_dice = []
            for i in range(1, myops2020.num_classes):
                val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :], activation=None))

            mean_dice = sum(val_class_dice) / len(val_class_dice)
            val_class_dices += np.array(val_class_dice)
            print('mean: {:.4} - lvm: {:.4} - lv: {:.4} - rv: {:.4} - lvme: {:.4} - lvms: {:.4}'
                  .format(mean_dice, val_class_dice[0], val_class_dice[1], val_class_dice[2], val_class_dice[3],
                          val_class_dice[4]))

        val_loss = np.average(val_losses)
        val_class_dices = val_class_dices / val_batch
        val_mean_dice = val_class_dices.sum() / val_class_dices.size
        lesion_mean_dice = (val_class_dices[3] + val_class_dices[4]) / 2

        val_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        val_writer.add_scalar('main_loss', val_loss, epoch)
        val_writer.add_scalar('main_dice', val_mean_dice, epoch)
        val_writer.add_scalar('lesion_dice', lesion_mean_dice, epoch)

        print(
            'val_loss: {:.4} - val_mean_dice: {:.4} - mean: {:.4} - lvm: {:.4}- lv: {:.4} - rv: {:.4} - lvme: {:.4} - lvms: {:.4}'
            .format(val_loss, val_mean_dice, lesion_mean_dice, val_class_dices[0], val_class_dices[1],
                    val_class_dices[2],
                    val_class_dices[3], val_class_dices[4]))
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        if scheduler_type == 'StepLR':
            scheduler.step()
        elif scheduler_type == 'ReduceLR':
            scheduler.step(val_loss)
        elif scheduler_type == 'poly':
            scheduler.step(epoch)
        else:
            pass

        early_stopping(lesion_mean_dice, net, epoch)
        # if early_stopping.early_stop:
        if early_stopping.early_stop or optimizer.param_groups[0]['lr'] < threshold_lr:
            print("Early stopping")
            # 结束模型训练
            break

    print('----------------------------------------------------------')
    print('save epoch {}'.format(early_stopping.save_epoch))
    print('stoped epoch {}'.format(epoch))
    print('----------------------------------------------------------')


if __name__ == '__main__':
    main()
