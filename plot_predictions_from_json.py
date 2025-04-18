import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms import functional as F
from PIL import Image
import os

from dataset import CocoCrackDataset
from model import Model6, pretrainedModel  # or Model2, Model1, etc.
from dl_utils import get_transform

def visualize_predictions(model, dataset, device, num_images=5, threshold=0.5, output_dir="viz_results"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for idx in range(num_images):
        img, target = dataset[idx]
        img_tensor = img.to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        img_pil = F.to_pil_image(img_tensor.cpu())
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_pil)

        # Ground truth boxes in GREEN
        for box in target['boxes']:
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        # Predicted boxes in RED
        for box, score in zip(prediction['boxes'], prediction['scores']):
            if score >= threshold:
                xmin, ymin, xmax, ymax = box.cpu().numpy()
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin - 5, f"{score:.2f}", color="red", fontsize=8, backgroundcolor="white")

        plt.axis('off')
        plt.title(f"Image {idx+1}: Green=GT, Red=Pred ≥ {threshold}")

        green_patch = patches.Patch(color='green', label='Ground Truth')
        red_patch = patches.Patch(color='red', label='Prediction')
        ax.legend(handles=[green_patch, red_patch], loc='lower right')

        save_path = os.path.join(output_dir, f"prediction_{idx+1:03d}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CocoCrackDataset(
        root="dataset/test/images",
        annotation="dataset/test/_annotations.coco.json",
        transforms=get_transform(train=False)
    )

    # Load model
    model = pretrainedModel(num_classes=2)
    model.load_state_dict(torch.load("model_PreTrained.pth", map_location=device))
    model.to(device)

    visualize_predictions(model, dataset, device, num_images=10, threshold=0.15, output_dir="viz_results")
