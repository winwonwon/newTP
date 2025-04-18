# dl_utils.py

import torchvision.transforms as T

def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms += [
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.2),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomRotation(degrees=10),
            T.RandomAdjustSharpness(2, p=0.3),
            T.RandomAutocontrast(p=0.2)
        ]
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))
