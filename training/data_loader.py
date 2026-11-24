from torch.utils.data import DataLoader

from dataset.dataset_loader import create_datasets

from training.data_transforms import (
    train_transform,
    validation_transform
)

from training.training_config import (
    BATCH_SIZE,
    NUM_WORKERS
)


# ---------------------------------------------------------
# CREATE DATASETS
# ---------------------------------------------------------

train_dataset, validation_dataset, test_dataset = create_datasets(
    train_transform=train_transform,
    validation_transform=validation_transform
)


# ---------------------------------------------------------
# CREATE DATALOADERS
# ---------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)