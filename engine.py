# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        tensorss, masks = samples.decompose()
        #print("\ntensorss",tensorss.shape,"masks",masks.shape,"len(targets)",len(targets),"targets",targets)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print("\nlen(targets)",len(targets),"devicetargets",targets)
        meta_info = {
            'size': torch.stack([t['size'][[1,0]] for t in targets]),  # (bs, 2)  W, H
        }
        #print("meta_info",meta_info)
        # tensorss torch.Size([2, 3, 649, 719]) masks torch.Size([2, 649, 719]) len(targets) 2 targets [{'boxes': tensor([[0.5135, 0.6438, 0.5922, 0.7079],
        #         [0.9075, 0.2632, 0.1850, 0.3718],
        #         [0.6180, 0.5522, 0.4735, 0.8957],
        #         [0.4745, 0.4449, 0.3493, 0.8225],
        #         [0.7242, 0.4787, 0.2519, 0.3641]], device='cuda:0'), 'labels': tensor([ 4,  1,  1,  1, 31], device='cuda:0'), 'image_id': tensor([122863], device='cuda:0'), 'area': tensor([55191.2070, 12934.5000, 37989.3086, 42231.8359,  3185.3691],
        #     device='cuda:0'), 'iscrowd': tensor([0, 0, 0, 0, 0], device='cuda:0'), 'orig_size': tensor([427, 640], device='cuda:0'), 'size': tensor([480, 719], device='cuda:0')}, {'boxes': tensor([[0.0085, 0.9647, 0.0171, 0.0707]], device='cuda:0'), 'labels': tensor([1], device='cuda:0'), 'image_id': tensor([561289], device='cuda:0'), 'area': tensor([476.3877], device='cuda:0'), 'iscrowd': tensor([0], device='cuda:0'), 'orig_size': tensor([640, 386], device='cuda:0'), 'size': tensor([649, 608], device='cuda:0')}]

        # len(targets) 2 devicetargets [{'boxes': tensor([[0.5135, 0.6438, 0.5922, 0.7079],
        #         [0.9075, 0.2632, 0.1850, 0.3718],
        #         [0.6180, 0.5522, 0.4735, 0.8957],
        #         [0.4745, 0.4449, 0.3493, 0.8225],
        #         [0.7242, 0.4787, 0.2519, 0.3641]], device='cuda:0'), 'labels': tensor([ 4,  1,  1,  1, 31], device='cuda:0'), 'image_id': tensor([122863], device='cuda:0'), 'area': tensor([55191.2070, 12934.5000, 37989.3086, 42231.8359,  3185.3691],
        #     device='cuda:0'), 'iscrowd': tensor([0, 0, 0, 0, 0], device='cuda:0'), 'orig_size': tensor([427, 640], device='cuda:0'), 'size': tensor([480, 719], device='cuda:0')}, {'boxes': tensor([[0.0085, 0.9647, 0.0171, 0.0707]], device='cuda:0'), 'labels': tensor([1], device='cuda:0'), 'image_id': tensor([561289], device='cuda:0'), 'area': tensor([476.3877], device='cuda:0'), 'iscrowd': tensor([0], device='cuda:0'), 'orig_size': tensor([640, 386], device='cuda:0'), 'size': tensor([649, 608], device='cuda:0')}]
        # meta_info {'size': tensor([[719, 480],
        #         [608, 649]], device='cuda:0')}


        outputs = model(samples, meta_info)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import numpy as np
from terminaltables import AsciiTable
import itertools

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        meta_info = {
            'size': torch.stack([t['size'][[1,0]] for t in targets]),  # (bs, 2)  W, H
        }

        outputs = model(samples, meta_info)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        #print("\ntaregts",targets, "\noutputs:",outputs)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        #print("\npostprocessors outputs",outputs)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        #print("\npostprocessors outputs",results)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        #print("\ntaregts",targets, "\noutputs:",outputs)
        #print("\n*********************coco_evaluator res:",res)
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        # *******add class wise
        coco_eva = coco_evaluator.coco_eval['bbox']
        precisions = coco_eva.eval['precision']
        #print("precisions",precisions.shape)
        #precisions (10, 101, 12, 4, 3)
        # precision: (iou, recall, cls, area range, max dets)
        coco_true = coco_eva.cocoGt
        cat_ids = sorted(coco_true.getCatIds())
        assert len(cat_ids) == precisions.shape[2]

        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_true.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (f'{nm["name"]}', f'{float(ap):0.3f}'))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(
            itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns]
            for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print('\n' + table.table)


    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
