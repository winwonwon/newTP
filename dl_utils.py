# dl_utils.py

import torchvision.transforms as T

def get_transform(train):
    if train:
        transforms = [
            # PIL-based transforms first
            T.RandomGrayscale(p=0.2),  # 20% chance to convert to grayscale
            T.ColorJitter(brightness=0.3, contrast=0.3),  # Optional: keep color jitter
            
            # Tensor conversions and subsequent transforms
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transforms = [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def collate_fn(batch):
    return tuple(zip(*batch))