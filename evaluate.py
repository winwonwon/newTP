# evaluate.py

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
import torch
from tqdm import tqdm

def evaluate_model(model, dataset, annotation_file, device, results_file='results.json'):
    model.eval()
    coco_gt = COCO(annotation_file)
    results = []

    for idx in tqdm(range(len(dataset))):
        img, target = dataset[idx]
        img_id = target['image_id'].item()
        img_tensor = img.to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            result = {
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [xmin.item(), ymin.item(), width.item(), height.item()],
                'score': score.item()
            }
            results.append(result)

    # Save results to JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Load results and evaluate
    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
