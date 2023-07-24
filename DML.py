#!/bin/env python
import os
import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

cudnn.benchmark = True

from torch.autograd import Variable
from timm.scheduler.cosine_lr import CosineLRScheduler
from dataset import get_dataset
from models import cifar_model_dict
from config import cfg
from distillers._base import Vanilla
from distillers.DKD import dkd_loss 
from distillers.myKD import myKD
from distillers.normKD import normKD
from util import validate, adjust_learning_rate, cross_validate, AverageMeter
from mix_util import mixup_data, mix2up_data, cutmix_data, cutmix_mixup_data
from models.discriminator import discriminator

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def kd_loss_weight(logits_student, logits_teacher, temperature, weight):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd = loss_kd * weight
    loss_kd = loss_kd.mean()
    loss_kd *= temperature**2
    return loss_kd


def cross_distillation(logits_student, logits_teacher, target):
    batch_size = target.size(0)
    cls_student = F.cross_entropy(logits_student, target, reduction='none').detach()
    cls_teacher = F.cross_entropy(logits_teacher, target, reduction='none').detach()
    weight = F.softmax(torch.stack([cls_student, cls_teacher]), dim=0)[0]
    # for i in range(batch_size):
    #     loss += weight[i] * kd_loss(logits_student[None, i], logits_teacher[None, i], 4.0)
    loss = kd_loss_weight(logits_student, logits_teacher, 4.0, weight)
    return loss

def main(cfg):
    train_loader, test_loader, num_data, num_classes = get_dataset(cfg.DATASET, cfg.OPTIMIZER.BATCH_SIZE,cfg.OPTIMIZER.BATCH_SIZE,8)

    net1, _ = cifar_model_dict[cfg.MODEL.MODEL1]
    net2, _ = cifar_model_dict[cfg.MODEL.MODEL2]
    model1 = net1(num_classes=num_classes)
    model2 = net2(num_classes=num_classes)

    state_dict1 = torch.load(cfg.MODEL.MODEL1_DIR, map_location="cpu")["model"]
    state_dict2 = torch.load(cfg.MODEL.MODEL2_DIR, map_location="cpu")["model"]
    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)

    if torch.cuda.is_available():
        model1 = model1.cuda()
        model2 = model2.cuda()
        model1 = torch.nn.DataParallel(model1)
        model2 = torch.nn.DataParallel(model2)

    
    if cfg.OPTIMIZER.TYPE == "SGD":
        optimizer1 = torch.optim.SGD(
            model1.parameters(),
            lr=cfg.OPTIMIZER.LR * 0.01,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
        optimizer2 = torch.optim.SGD(
            model2.parameters(),
            lr=cfg.OPTIMIZER.LR * 0.01,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.TYPE == "Adam":
        optimizer1 = torch.optim.AdamW(
            model1.parameters(),
            lr=cfg.OPTIMIZER.LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
        optimizer2 = torch.optim.Adam(
            model2.parameters(),
            lr=cfg.OPTIMIZER.LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer,
    #    mode='max',
    #    factor=0.316,
    #    patience=20,
    #    min_lr=1e-6
    # )
    best_acc_model1 = 0
    best_acc_model2 = 0
    best_acc5_model1 = 0
    best_acc5_model2 = 0
    loss1_avg = AverageMeter()
    loss2_avg = AverageMeter()


    for epoch in range(cfg.OPTIMIZER.EPOCHS):
        lr = adjust_learning_rate(optimizer1, epoch, cfg) 
        lr = adjust_learning_rate(optimizer2, epoch, cfg) 

        print("Epoch: {}, LR: {}".format(epoch, optimizer1.param_groups[0]["lr"]))
        model1.train()
        model2.train()
        pbar = tqdm.tqdm(range(len(train_loader)))
        for i, (img, target, index) in enumerate(train_loader):
            img, target = Variable(img), Variable(target)
            img = img.cuda()
            target = target.cuda()

            # loss1 = torch.tensor(0.0).cuda()
            # loss1_cls = nn.CrossEntropyLoss()(logits1, target)
            # loss1_div = min(epoch/20, 1.0) * 0.01 * cross_distillation(logits1, Variable(logits2), target)
            # loss1_div = min(epoch/20, 1.0) * dkd_loss(logits1, logits2, target, 1.0, 2.0, 4.0) 
            # loss1 = loss1_cls + loss1_div 
            
            # loss1, losses1 = mix_distillation(model1, model2, img, target, epoch=epoch, alpha=1, mix_method=cutmix_data, cfg=cfg)
            logit1, _ = model1(img)
            logit2, _ = model2(img)

            loss1 = cross_distillation(logit1, logit2.detach(), target)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = cross_distillation(logit2, logit1.detach(), target)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            loss1_avg.update(loss1.item())
            loss2_avg.update(loss2.item())

            pbar.update()
            pbar.set_description(f"Loss1: {loss1_avg.avg:.4f}, Loss2: {loss2_avg.avg:.4f}")
            
        pbar.close()
        loss1_avg.reset()
        loss2_avg.reset()

        model1.eval()
        model2.eval()
        test_acc_model1, test_acc5_model1, test_loss_model1 = cross_validate(test_loader, model1)
        test_acc_model2, test_acc5_model2, test_loss_model2 = cross_validate(test_loader, model2)

        if test_acc_model1 > best_acc_model1:
            best_acc_model1 = test_acc_model1
            state = {
                "model": model1.module.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc_model1,
                "optimizer": optimizer1.state_dict(),
            }
            torch.save(state, cfg.SAVE_DIR+"DML_run{}_model1.pth".format(cfg.MODEL.COUNT))
        if test_acc_model2 > best_acc_model2:
            best_acc_model2 = test_acc_model2
            state = {
                "model": model2.module.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc_model2,
                "optimizer": optimizer2.state_dict(),
            }
            torch.save(state, cfg.SAVE_DIR+"DML_run{}_model2.pth".format(cfg.MODEL.COUNT))
        if test_acc5_model1 > best_acc5_model1:
            best_acc5_model1 = test_acc5_model1
        if test_acc5_model2 > best_acc5_model2:
            best_acc5_model2 = test_acc5_model2

        print(f"[Model 1]Best Acc: {best_acc_model1:.4f}, Best Acc5: {best_acc5_model1:.4f}")
        print(f"[Model 2]Best Acc: {best_acc_model2:.4f}, Best Acc5: {best_acc5_model2:.4f}")
        # scheduler.step(test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, default='config_cross.yaml')
    parser.add_argument('-count', type=int, default=0)
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.MODEL.COUNT = args.count
    main(cfg)

