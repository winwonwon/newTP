# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CocoCrackDataset
from dl_utils import get_transform, collate_fn
from model import get_model_instance
from evaluate import evaluate_model
from visualize import visualize_predictions
import os

def load_datasets(data_dir):
    train_ds = CocoCrackDataset(
        root=os.path.join(data_dir, "train/images"),
        annotation=os.path.join(data_dir, "train/_annotations.coco.json"),
        transforms=get_transform(train=True)
    )
    val_ds = CocoCrackDataset(
        root=os.path.join(data_dir, "valid/images"),
        annotation=os.path.join(data_dir, "valid/_annotations.coco.json"),
        transforms=get_transform(train=False)
    )
    test_ds = CocoCrackDataset(
        root=os.path.join(data_dir, "test/images"),
        annotation=os.path.join(data_dir, "test/_annotations.coco.json"),
        transforms=get_transform(train=False)
    )
    return train_ds, val_ds, test_ds

def train_model(data_dir, num_epochs=10, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = load_datasets(data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model_instance(num_classes=2)  # Cracks + background
    model.to(device)

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001, momentum=0.9, weight_decay=0.0005
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")

    # Evaluate on test set
    annotation_file = os.path.join(data_dir, "test/_annotations.coco.json")
    evaluate_model(model, test_ds, annotation_file, device)

    # Visualize predictions on test set
    visualize_predictions(model, test_ds, device, num_images=5)

if __name__ == "__main__":
    data_path = "dataset"
    train_model(data_path,1,8)
