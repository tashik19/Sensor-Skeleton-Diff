import os
import torch
import random
import numpy as np
import torch.nn as nn
from .model import Diffusion1D
from .sensor_model import CombinedLSTMClassifier

NUM_CLASSES  = 12
HIDDEN_SIZE  = 128   # was 256 — halved
NUM_LAYERS   = 3     # was 8 — reduced significantly
CONV_CH      = 32    # was 16 — increase conv capacity to compensate
KERNEL_SIZE  = 3
DROPOUT      = 0.3   # was 0.5 — reduce dropout since model is smaller
NUM_HEADS    = 4


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def initialize_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def load_sensor_model(args, device):
    set_seed(args.seed)

    window_size = getattr(args, 'window_size', 90)

    model = CombinedLSTMClassifier(
        sensor_input_size=3,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        conv_channels=CONV_CH,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        window_size=window_size,     # ← must match dataset window_size
    ).to(device)

    model.apply(initialize_weights)

    if not args.train_sensor_model:
        path = os.path.join(args.output_dir, "sensor_model", "best_sensor_model.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            if any(k.startswith("module.") for k in ckpt):
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            model.eval()
            print(f"Loaded sensor model from {path}")
        else:
            raise FileNotFoundError(f"No pre-trained sensor model at {path}")

    return model


def load_diffusion(device, skeleton_dim=48, num_classes=NUM_CLASSES):
    return Diffusion1D(skeleton_dim=skeleton_dim, num_classes=num_classes).to(device)


def load_diffusion_model_for_testing(
    device, output_dir, test_diffusion_model,
    skeleton_dim=48, num_classes=NUM_CLASSES
):
    model = Diffusion1D(skeleton_dim=skeleton_dim, num_classes=num_classes).to(device)
    path  = os.path.join(output_dir, "diffusion_model", "best_diffusion_model.pth")
    print(f"Checkpoint path: {path}")

    if test_diffusion_model:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            if any(k.startswith("module.") for k in ckpt):
                print("Removing 'module.' prefix...")
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            print(f"Loaded diffusion model from {path}")
        else:
            print("No checkpoint found. Initializing new model.")
    else:
        print("Initializing new diffusion model.")

    return model
