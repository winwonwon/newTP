import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CocoCrackDataset
from dl_utils import get_transform, collate_fn
from model import pretrainedModel, Model1, Model2, Model3, Model4, Model5, Model6
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

def train_model(data_dir, num_epochs=10, batch_size=4, model_choice="Model6"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = load_datasets(data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Select model
    model_dict = {
        "Model1": Model1,
        "Model2": Model2,
        "Model6": Model6
    }
    model = model_dict[model_choice](num_classes=2)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"\nTraining {model_choice} for {num_epochs} epochs...\n")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        elapsed = time.time() - start_time

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {elapsed:.2f}s")

    # === Evaluate and visualize on validation set ===
    print("\nEvaluating on validation set...")
    val_ann_file = os.path.join(data_dir, "valid/_annotations.coco.json")
    os.makedirs("outputs/val", exist_ok=True)
    evaluate_model(model, val_ds, val_ann_file, device, results_file="outputs/val/results.json")
    visualize_predictions(model, val_ds, device, output_dir="outputs/val/visualizations")

    # === Evaluate and visualize on test set ===
    print("\nEvaluating on test set...")
    test_ann_file = os.path.join(data_dir, "test/_annotations.coco.json")
    os.makedirs("outputs/test", exist_ok=True)
    evaluate_model(model, test_ds, test_ann_file, device, results_file="outputs/test/results.json")
    visualize_predictions(model, test_ds, device, output_dir="outputs/test/visualizations")

    # Save model
    model_file = f"model_{model_choice}.pth"
    torch.save(model.state_dict(), model_file)
    print(f"\nModel saved to {model_file}")

if __name__ == "__main__":
    data_path = "dataset"
    train_model(data_path, num_epochs=15, batch_size=4, model_choice="Model1")