#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch
from utilities.utils import AverageMeter
import time
from utilities.metrics.segmentation_miou import MIOU
from utilities.print_utils import *
from torch.nn.parallel import gather
from loss_fns.segmentation_loss import SegmentationLoss, NIDLoss, PixelwiseKLD
from collections import OrderedDict

def train_seg(model, dataset_loader, optimizer, criterion, num_classes, epoch, device='cuda', use_depth=False, add_criterion=None, weight=1.0):
    losses = AverageMeter()
    ce_losses = AverageMeter()
    nid_losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    end = time.time()
    model.train()

    miou_class = MIOU(num_classes=num_classes-1)
    for i, batch in enumerate(dataset_loader):
        inputs = batch[0].to(device=device)
        target = batch[1].to(device=device)

        if use_depth:
            depth = batch[2].to(device=device)
            outputs = model(inputs, depth)
        else:
            outputs = model(inputs)

        if isinstance(outputs, OrderedDict):
            outputs = outputs['out']

        if device == 'cuda':
#            print("Target size {}".format(target.size()))
#
            loss = criterion(outputs, target).mean()
            if add_criterion is not None:
                loss2 = add_criterion(inputs, outputs.to(device)) * weight
                loss += loss2

            if isinstance(outputs, (list, tuple)):
                target_dev = outputs[0].device
                outputs = gather(outputs, target_device=target_dev)
        else:
            loss = criterion(outputs, target)
            if add_criterion is not None:
                loss2 = add_criterion(inputs, outputs) * weight
                loss += loss2

        inter, union = miou_class.get_iou(outputs, target)

        inter_meter.update(inter)
        union_meter.update(union)
        
        losses.update(loss.item(), inputs.size(0))
        if add_criterion is not None:
            nid_losses.update(loss2.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:  # print after every 100 batches
            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = iou.mean() * 100
            print_log_message("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f\t\tNID loss:%.4f" %
                  (epoch, i, len(dataset_loader), batch_time.avg, losses.avg, miou, nid_losses.avg))

    print(inter_meter.sum)
    print(union_meter.sum)
    print(inter_meter.sum / (union_meter.sum + 1e-10) * 100)
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    return miou, losses.avg

def val_seg(model, dataset_loader, criterion=None, num_classes=21, device='cuda', use_depth=False, add_criterion=None):
    model.eval()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    miou_class = MIOU(num_classes=num_classes-1)

    if criterion:
        losses = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(dataset_loader):
            inputs = batch[0].to(device=device)
            target = batch[1].to(device=device)
            
            if use_depth:
                depth = batch[2].to(device=device)
                outputs = model(inputs, depth)
            else:
                outputs = model(inputs)

            if isinstance(outputs, OrderedDict):
                outputs = outputs['out']

            if criterion:
                if device == 'cuda':
                    loss = criterion(outputs, target).mean()
                    if add_criterion is not None:
                        loss += add_criterion(inputs, outputs)
                    if isinstance(outputs, (list, tuple)):
                        target_dev = outputs[0].device
                        outputs = gather(outputs, target_device=target_dev)
                else:
                    loss = criterion(outputs, target)
                    if add_criterion is not None:
                        loss += add_criterion(inputs, outputs)

                losses.update(loss.item(), inputs.size(0))

            inter, union = miou_class.get_iou(outputs, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:  # print after every 100 batches
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = iou.mean() * 100
                loss_ = losses.avg if criterion is not None else 0
                print_log_message("[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                      (i, len(dataset_loader), batch_time.avg, loss_, miou))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    print_info_message('Mean IoU: {0:.2f}'.format(miou))
    if criterion:
        return miou, losses.avg
    else:
        return miou, 0

def train_seg_ue(model, dataset_loader, optimizer, criterion, num_classes, 
                 epoch, device='cuda', use_depth=False, add_criterion=None, weight=1.0):
    losses = AverageMeter()
    ce_losses = AverageMeter()
    nid_losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    end = time.time()
    model.train()

    miou_class = MIOU(num_classes=num_classes-1)
    kld_layer = PixelwiseKLD()
    print("train_seg_ue()")
    for i, batch in enumerate(dataset_loader):
        inputs = batch[0].to(device=device)
        target = batch[1].to(device=device)

        if use_depth:
            depth = batch[2].to(device=device)
            outputs = model(inputs, depth)
        else:
            outputs = model(inputs)

        if isinstance(outputs, OrderedDict):
            out_aux = outputs['aux']
            outputs = outputs['out']
        else:
            out_aux = outputs[1]
            outputs = outputs[0]

        kld = kld_layer(outputs, out_aux)
        outputs = outputs + 0.5*out_aux
        
        if device == 'cuda':
#            print("Target size {}".format(target.size()))
#
            loss = criterion(outputs, target).mean() # + kld.mean()
            if add_criterion is not None:
                loss2 = add_criterion(inputs, outputs.to(device)) * weight
                loss += loss2            

            if isinstance(outputs, (list, tuple)):
                target_dev = outputs[0].device
                outputs = gather(outputs, target_device=target_dev)
        else:
            loss = criterion(outputs, target)# + kld.mean()
            if add_criterion is not None:
                loss2 = add_criterion(inputs, outputs) * weight
                loss += loss2

        inter, union = miou_class.get_iou(outputs, target)

        inter_meter.update(inter)
        union_meter.update(union)
        
        losses.update(loss.item(), inputs.size(0))
        if add_criterion is not None:
            nid_losses.update(loss2.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:  # print after every 100 batches
            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = iou.mean() * 100
            print_log_message("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f\t\tNID loss:%.4f" %
                  (epoch, i, len(dataset_loader), batch_time.avg, losses.avg, miou, nid_losses.avg))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100
    print(inter_meter.sum)
    print(union_meter.sum)
    print(inter_meter.sum / (union_meter.sum + 1e-10) * 100)

    return miou, losses.avg

def val_seg_ue(model, dataset_loader, criterion=None, num_classes=21, device='cuda', use_depth=False, add_criterion=None):
    model.eval()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    miou_class = MIOU(num_classes=num_classes-1)

    if criterion:
        losses = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(dataset_loader):
            inputs = batch[0].to(device=device)
            target = batch[1].to(device=device)
            
            if use_depth:
                depth = batch[2].to(device=device)
                outputs = model(inputs, depth)
            else:
                outputs = model(inputs)

            if isinstance(outputs, OrderedDict):
                out_aux = outputs['aux']
                outputs = outputs['out']
            else:
                out_aux = outputs[1]
                outputs = outputs[0]

            outputs = outputs + 0.5 * out_aux

            if criterion:
                if device == 'cuda':
                    loss = criterion(outputs, target).mean()
                    if add_criterion is not None:
                        loss += add_criterion(inputs, outputs)
                    if isinstance(outputs, (list, tuple)):
                        target_dev = outputs[0].device
                        outputs = gather(outputs, target_device=target_dev)
                else:
                    loss = criterion(outputs, target)
                    if add_criterion is not None:
                        loss += add_criterion(inputs, outputs)

                losses.update(loss.item(), inputs.size(0))

            inter, union = miou_class.get_iou(outputs, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:  # print after every 100 batches
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = iou.mean() * 100
                loss_ = losses.avg if criterion is not None else 0
                print_log_message("[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                      (i, len(dataset_loader), batch_time.avg, loss_, miou))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    print_info_message('Mean IoU: {0:.2f}'.format(miou))
    if criterion:
        return miou, losses.avg
    else:
        return miou, 0

