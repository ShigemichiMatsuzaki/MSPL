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

def train_seg_cls(model, dataset_loader, optimizer, criterion_seg, num_classes, epoch, criterion_cls, cls_loss_weight=1.0, device='cuda', use_depth=False):
    losses = AverageMeter()
    cls_losses = AverageMeter()
    seg_losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    end = time.time()
    model.train()

    miou_class = MIOU(num_classes=num_classes)

    for i, batch in enumerate(dataset_loader):
        inputs = batch[0].to(device=device)
        target = batch[1].to(device=device)

        if use_depth:
            depth = batch[2].to(device=device)
            outputs_seg, outputs_cls = model(inputs, depth)
        else:
            outputs_seg, outputs_cls = model(inputs)

        cls_ids = batch[3].to(device=device)

        if device == 'cuda':
            loss_seg = criterion_seg(outputs_seg, target).mean()

            loss_cls = criterion_cls(outputs_cls, cls_ids).mean()
            loss = loss_seg + cls_loss_weight * loss_cls


            if isinstance(outputs_seg, (list, tuple)):
                target_dev = outputs[0].device
                outputs_seg = gather(outputs_seg, target_device=target_dev)
        else:
            loss_seg = criterion_seg(outputs_seg, target)

            loss_cls = criterion_cls(outputs_cls, cls_ids)
            loss = loss_seg + cls_loss_weight * loss_cls

        inter, union = miou_class.get_iou(outputs_seg, target)

        inter_meter.update(inter)
        union_meter.update(union)

        losses.update(loss.item(), inputs.size(0))
        seg_losses.update(loss_seg.item(), inputs.size(0))
        cls_losses.update(loss_cls.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:  # print after every 100 batches
            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = iou.mean() * 100
            print_log_message("Epoch: %d[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                  (epoch, i, len(dataset_loader), batch_time.avg, losses.avg, miou))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    return miou, losses.avg, seg_losses.avg, cls_losses.avg


def val_seg_cls(model, dataset_loader, criterion_seg=None, criterion_cls=None, num_classes=21, cls_loss_weight=1.0, device='cuda', use_depth=False):
    model.eval()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    miou_class = MIOU(num_classes=num_classes)

    if criterion_seg:
        losses = AverageMeter()
        cls_losses = AverageMeter()
        seg_losses = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(dataset_loader):
            inputs = batch[0].to(device=device)
            target = batch[1].to(device=device)
            
            if use_depth:
                depth = batch[2].to(device=device)
                outputs_seg, outputs_cls = model(inputs, depth)
            else:
                outputs_seg, outputs_cls = model(inputs)

            cls_ids = batch[3].to(device=device)

            if criterion_seg and criterion_cls:
                if device == 'cuda':
                    loss_seg = criterion_seg(outputs_seg, target).mean()
                    loss_cls = criterion_cls(outputs_cls, cls_ids).mean()

                    loss = loss_seg + cls_loss_weight * loss_cls

                    if isinstance(outputs_seg, (list, tuple)):
                        target_dev = outputs[0].device
                        outputs_seg = gather(outputs_seg, target_device=target_dev)
                else:
                    loss_seg = criterion_seg(outputs_seg, target)
                    loss_cls = criterion_cls(outputs_cls, cls_ids)

                    loss = loss_seg + cls_loss_weight * loss_cls

                losses.update(loss.item(), inputs.size(0))
                seg_losses.update(loss_seg.item(), inputs.size(0))
                cls_losses.update(loss_cls.item(), inputs.size(0))

            inter, union = miou_class.get_iou(outputs_seg, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:  # print after every 100 batches
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                miou = iou.mean() * 100
                loss_ = losses.avg if criterion_seg is not None else 0
                print_log_message("[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
                      (i, len(dataset_loader), batch_time.avg, loss_, miou))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    print_info_message('Mean IoU: {0:.2f}'.format(miou))
    if criterion_seg and criterion_cls:
        return miou, losses.avg, seg_losses.avg, cls_losses.avg
    else:
        return miou, 0, 0, 0
