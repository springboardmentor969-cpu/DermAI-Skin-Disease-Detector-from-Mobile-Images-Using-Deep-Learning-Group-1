import torch
import torch.nn as nn

from training.class_weights import calculate_class_weights
from training.training_config import LEARNING_RATE, WEIGHT_DECAY


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# CLASS-WEIGHTED LOSS
# ---------------------------------------------------------

def create_loss_function():

    _, class_weights = calculate_class_weights()

    class_weights = class_weights.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion


# ---------------------------------------------------------
# OPTIMIZER
# ---------------------------------------------------------

def create_optimizer(model):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    return optimizer