# visualize.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms import functional as F
from PIL import Image
import os

def visualize_predictions(model, dataset, device, num_images=5, threshold=0.5):
    model.eval()
    for idx in range(num_images):
        img, target = dataset[idx]
        img_tensor = img.to(device)
        with torch.no_grad():
            prediction = model([img_tensor])[0]

        img_np = F.to_pil_image(img_tensor.cpu())
        fig, ax = plt.subplots(1)
        ax.imshow(img_np)

        # Ground truth boxes
        for box in target['boxes']:
            xmin, ymin, xmax, ymax = box.cpu()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        # Predicted boxes
        for box, score in zip(prediction['boxes'], prediction['scores']):
            if score >= threshold:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

        plt.title(f"Image {idx+1}: Green=GT, Red=Pred")
        plt.axis('off')
        plt.show()
