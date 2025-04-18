# plot_predictions_from_json.py

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from PIL import Image

def plot_coco_predictions(results_file, annotation_file, image_dir, output_dir="viz_results", max_images=10, threshold=0.7):
    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth annotations
    coco_gt = COCO(annotation_file)

    # Load prediction results
    with open(results_file) as f:
        preds = json.load(f)

    # Group predictions by image_id with threshold filter
    image_pred_map = {}
    for p in preds:
        if p['score'] >= threshold:
            image_pred_map.setdefault(p['image_id'], []).append(p)

    # Visualize predictions for each image
    for i, img_id in enumerate(image_pred_map):
        if i >= max_images:
            break

        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(image_dir, file_name)

        image = Image.open(img_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        # Ground Truth Boxes (Green)
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        # Predicted Boxes (Red)
        for pred in image_pred_map[img_id]:
            x, y, w, h = pred['bbox']
            score = pred['score']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 5, f"{score:.2f}", color="red", fontsize=8, backgroundcolor="white")

        plt.axis('off')
        plt.title(f"{file_name} (Green = GT, Red = Pred ≥ {threshold})")

        # Add legend
        green_patch = patches.Patch(color='green', label='Ground Truth')
        red_patch = patches.Patch(color='red', label='Prediction')
        ax.legend(handles=[green_patch, red_patch], loc='lower right')

        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    results_file = "results.json"
    annotation_file = "dataset/test/_annotations.coco.json"
    image_dir = "dataset/test/images"
    plot_coco_predictions(results_file, annotation_file, image_dir, threshold=0.55)
