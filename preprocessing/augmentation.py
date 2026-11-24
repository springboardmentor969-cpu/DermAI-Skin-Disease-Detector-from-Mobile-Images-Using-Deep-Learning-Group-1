import torchvision.transforms as transforms


# ============================================================
# DERMAI DATA AUGMENTATION
# ============================================================

def get_train_transforms():
    """
    Augmentation applied only to training images.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),

        # Small geometric variations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=15,
            fill=0
        ),

        # Simulate small scale variations
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.90, 1.0),
            ratio=(0.95, 1.05)
        ),

        # Small illumination/color variations
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),

        # Convert to tensor
        transforms.ToTensor(),

        # ImageNet normalization for pretrained CNNs
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_validation_transforms():
    """
    Validation images are NOT augmented.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transforms():
    """
    Test images are NOT augmented.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])