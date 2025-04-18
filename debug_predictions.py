import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from dataset import CocoCrackDataset
from model import Model2  # or Model6, etc.
from dl_utils import get_transform

def debug_predictions(model, dataset, device, threshold=0.001, num_images=5):
    model.eval()

    for idx in range(num_images):
        img, target = dataset[idx]
        img_tensor = img.to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        img_pil = F.to_pil_image(img_tensor.cpu())
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_pil)

        # GT boxes
        for box in target['boxes']:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        print(f"\nImage {idx + 1} | GT labels: {target['labels'].tolist()}")

        # Predictions
        for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            if score >= threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"{score:.2f} (label {label})", color='red', fontsize=8)

                print(f"→ Detected label {label.item()} with score {score.item():.3f}")

        plt.axis('off')
        plt.title(f"Image {idx + 1}: Green=GT, Red=Preds ≥ {threshold}")
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CocoCrackDataset(
        root="dataset/valid/images",
        annotation="dataset/valid/_annotations.coco.json",
        transforms=get_transform(train=False)
    )

    model = Model2(num_classes=2)
    model.load_state_dict(torch.load("model_Model2.pth", map_location=device))
    model.to(device)

    debug_predictions(model, dataset, device, threshold=0.001, num_images=5)
