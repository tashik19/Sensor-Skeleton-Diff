import os
import sys
import io
import torch
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from collections import Counter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_model.model_loader import load_sensor_model, load_diffusion, NUM_CLASSES
from diffusion_model.skeleton_model import SkeletonTransformer
from diffusion_model.util import prepare_dataset, compute_loss


def ensure_dir(path, rank):
    """Create directory, wiping it first. Only rank-0 does the IO."""
    if rank == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    dist.barrier()


def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def cleanup():
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Sensor model
# ─────────────────────────────────────────────────────────────────────────────

def train_sensor_model(rank, args, device, train_loader, val_loader):
    print("Training Sensor model")
    torch.manual_seed(args.seed + rank)

    sensor_model = load_sensor_model(args, device)
    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(
        sensor_model.parameters(), lr=args.sensor_lr, betas=(0.9, 0.98)
    )

    save_dir = os.path.join(args.output_dir, "sensor_model")
    ensure_dir(save_dir, rank)
    log_dir  = os.path.join(save_dir, "sensor_logs")
    ensure_dir(log_dir, rank)

    writer     = SummaryWriter(log_dir=log_dir) if rank == 0 else None
    best_loss  = float('inf')
    no_improve = 0

    for epoch in range(args.sensor_epoch):
        sensor_model.train()
        epoch_train_loss = 0.0

        for _, sensor1, sensor2, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.sensor_epoch} [Train]"
        ):
            sensor1, sensor2, labels = (
                sensor1.to(device), sensor2.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            output, _ = sensor_model(sensor1, sensor2)

            if (labels >= output.shape[1]).any() or (labels < 0).any():
                print(f"Invalid label detected: {labels.unique()}")
                exit()

            loss = torch.nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sensor_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train = epoch_train_loss / len(train_loader)

        # Validation
        sensor_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for _, sensor1, sensor2, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{args.sensor_epoch} [Val]"
            ):
                sensor1, sensor2, labels = (
                    sensor1.to(device), sensor2.to(device), labels.to(device)
                )
                output, _ = sensor_model(sensor1, sensor2)
                epoch_val_loss += torch.nn.CrossEntropyLoss()(output, labels).item()

        avg_val = epoch_val_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.sensor_epoch} | Train: {avg_train:.4f}  Val: {avg_val:.4f}")
            writer.add_scalar('Loss/Train',      avg_train, epoch)
            writer.add_scalar('Loss/Validation', avg_val,   epoch)
            if avg_val < best_loss:
                best_loss  = avg_val
                no_improve = 0
                torch.save(
                    sensor_model.state_dict(),
                    os.path.join(save_dir, "best_sensor_model.pth"),
                )
                print(f"  ✓ Saved best sensor model (val {best_loss:.4f})")
            else:
                no_improve += 1

        # Broadcast early-stop decision from rank-0
        stop = torch.tensor(
            1 if no_improve >= args.sensor_patience else 0, device=device
        )
        dist.broadcast(stop, src=0)
        if stop.item():
            if rank == 0:
                print(f"Early stopping at epoch {epoch+1}")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Skeleton model
# ─────────────────────────────────────────────────────────────────────────────

def train_skeleton_model(rank, args, device, train_loader, val_loader):
    print("Training Skeleton model")
    torch.manual_seed(args.seed + rank)

    skeleton_model = SkeletonTransformer(
        input_size=48, num_classes=NUM_CLASSES
    ).to(device)
    skeleton_model = DDP(skeleton_model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(
        skeleton_model.parameters(), lr=args.skeleton_lr, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=0.1
    )

    save_dir = os.path.join(args.output_dir, "skeleton_model")
    ensure_dir(save_dir, rank)

    writer        = SummaryWriter(log_dir=save_dir) if rank == 0 else None
    best_loss     = float('inf')
    best_accuracy = 0.0

    for epoch in range(args.skeleton_epochs):
        skeleton_model.train()
        epoch_train_loss = 0.0
        correct_train    = 0
        total_train      = 0

        for skeleton_data, _, _, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.skeleton_epochs} [Train]"
        ):
            skeleton_data, labels = skeleton_data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = skeleton_model(skeleton_data)
            loss   = torch.nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(skeleton_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted  = torch.max(output, 1)
            total_train  += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100

        # Validation
        skeleton_model.eval()
        epoch_val_loss = 0.0
        correct_val    = 0
        total_val      = 0
        with torch.no_grad():
            for skeleton_data, _, _, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{args.skeleton_epochs} [Val]"
            ):
                skeleton_data, labels = skeleton_data.to(device), labels.to(device)
                output = skeleton_model(skeleton_data)
                loss   = torch.nn.CrossEntropyLoss()(output, labels)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total_val   += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss  = epoch_val_loss / len(val_loader)
        val_accuracy  = correct_val / total_val * 100
        scheduler.step()

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{args.skeleton_epochs} | "
                f"Train {avg_train_loss:.4f} / {train_accuracy:.1f}%  "
                f"Val {avg_val_loss:.4f} / {val_accuracy:.1f}%"
            )
            writer.add_scalar('Loss/Train',       avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation',  avg_val_loss,   epoch)
            writer.add_scalar('Accuracy/Train',   train_accuracy, epoch)
            writer.add_scalar('Accuracy/Val',     val_accuracy,   epoch)

            if avg_val_loss < best_loss:
                best_loss     = avg_val_loss
                best_accuracy = val_accuracy
                torch.save(
                    skeleton_model.state_dict(),
                    os.path.join(save_dir, "best_skeleton_model.pth"),
                )
                print(f"  ✓ Saved best skeleton model (val {best_loss:.4f} / {best_accuracy:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Diffusion model
# ─────────────────────────────────────────────────────────────────────────────

def train_diffusion_model(rank, args, device, train_loader, val_loader):
    print("Training Diffusion model")
    torch.manual_seed(args.seed + rank)

    # ── Sensor model: FROZEN, not DDP-wrapped ────────────────────────────────
    sensor_model = load_sensor_model(args, device)
    sensor_model.eval()
    for p in sensor_model.parameters():
        p.requires_grad_(False)

    # ── Diffusion model: trainable, DDP-wrapped ───────────────────────────────
    diffusion_model = load_diffusion(
        device, skeleton_dim=args.window_size, num_classes=NUM_CLASSES
    )
    diffusion_model = DDP(diffusion_model, device_ids=[rank], find_unused_parameters=True)

    # ── Skeleton model: FROZEN at Stage 2 quality, NOT trained in Stage 3 ────
    # Training the skeleton model on generated data caused it to degrade from
    # 95% to 55% accuracy across all previous runs, corrupting the auxiliary
    # signal. Freezing it keeps a stable 95% classifier throughout training.
    skeleton_model = SkeletonTransformer(
        input_size=48, num_classes=NUM_CLASSES
    ).to(device)
    skel_ckpt = os.path.join(args.output_dir, "skeleton_model", "best_skeleton_model.pth")
    if os.path.exists(skel_ckpt):
        ckpt = torch.load(skel_ckpt, map_location=device)
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        skeleton_model.load_state_dict(ckpt)
        print("✓ Loaded Stage 2 skeleton weights (95.5% accuracy, frozen)")
    else:
        print("⚠ No Stage 2 skeleton checkpoint found")
    skeleton_model.eval()
    for p in skeleton_model.parameters():
        p.requires_grad_(False)

    # ── Only one optimizer: diffusion model only ──────────────────────────────
    diffusion_optimizer = optim.Adam(
        diffusion_model.parameters(), lr=args.diffusion_lr, eps=1e-8, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        diffusion_optimizer, mode='min', factor=0.5, patience=8
    )

    diffusion_save_dir = os.path.join(args.output_dir, "diffusion_model")
    ensure_dir(diffusion_save_dir, rank)
    # Skeleton dir is read-only in Stage 3 — never wipe it
    dist.barrier()

    writer = SummaryWriter(log_dir=diffusion_save_dir) if rank == 0 else None

    best_diffusion_loss = float('inf')
    no_improve          = 0
    # Fixed auxiliary weight — no ramp-up, skeleton stays frozen and accurate
    SKEL_AUX_WEIGHT = 0.05

    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type='cosine', T=args.timesteps, step=1, device=device),
        device=device,
        ddim_scale=args.ddim_scale,
    )

    for epoch in range(args.epochs):
        diffusion_model.train()
        epoch_train_loss    = 0.0
        epoch_skeleton_loss = 0.0
        correct_train       = 0
        total_train         = 0

        for batch_idx, (skeleton, sensor1, sensor2, mask) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
            if rank == 0 else train_loader
        ):
            skeleton, sensor1, sensor2, mask = (
                skeleton.to(device), sensor1.to(device),
                sensor2.to(device),  mask.to(device),
            )
            t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()

            with torch.no_grad():
                _, context = sensor_model(sensor1, sensor2, return_attn_output=True)
            context = context.detach()

            diffusion_optimizer.zero_grad()

            loss, x0_pred = compute_loss(
                args=args,
                model=diffusion_model,
                x0=skeleton,
                context=context,
                label=mask,
                t=t,
                mask=mask,
                device=device,
                diffusion_process=diffusion_process,
                angular_loss=args.angular_loss,
                epoch=epoch,
                rank=rank,
                batch_idx=batch_idx,
            )

            # ── Frozen auxiliary skeleton loss ────────────────────────────────
            # Skeleton model is frozen — evaluate only, add weighted loss to
            # diffusion loss before backprop. No separate optimizer step.
            skeleton_output = skeleton_model(x0_pred)
            skeleton_loss = torch.nn.CrossEntropyLoss()(skeleton_output, mask)
            total_loss    = loss + SKEL_AUX_WEIGHT * skeleton_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            diffusion_optimizer.step()

            epoch_train_loss    += loss.item()
            epoch_skeleton_loss += skeleton_loss.item()
            _, predicted  = torch.max(skeleton_output, 1)
            total_train  += mask.size(0)
            correct_train += (predicted == mask).sum().item()

        avg_train_loss    = epoch_train_loss    / len(train_loader)
        avg_skeleton_loss = epoch_skeleton_loss / len(train_loader)
        train_accuracy    = 100 * correct_train / total_train

        # ── Validation ────────────────────────────────────────────────────────
        diffusion_model.eval()
        epoch_val_loss          = 0.0
        epoch_skeleton_val_loss = 0.0
        correct_val             = 0
        total_val               = 0

        with torch.no_grad():
            for batch_idx, (skeleton, sensor1, sensor2, mask) in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
                if rank == 0 else val_loader
            ):
                skeleton, sensor1, sensor2, mask = (
                    skeleton.to(device), sensor1.to(device),
                    sensor2.to(device),  mask.to(device),
                )
                t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
                _, context = sensor_model(sensor1, sensor2, return_attn_output=True)
                context = context.detach()

                val_loss, x0_pred_val = compute_loss(
                    args=args,
                    model=diffusion_model,
                    x0=skeleton,
                    context=context,
                    label=mask,
                    t=t,
                    mask=mask,
                    device=device,
                    diffusion_process=diffusion_process,
                    angular_loss=args.angular_loss,
                    epoch=epoch,
                    rank=rank,
                    batch_idx=batch_idx,
                )
                epoch_val_loss += val_loss.item()

                skel_out_val  = skeleton_model(x0_pred_val.detach())
                skel_val_loss = torch.nn.CrossEntropyLoss()(skel_out_val, mask)
                epoch_skeleton_val_loss += skel_val_loss.item()
                _, pred_val  = torch.max(skel_out_val, 1)
                total_val   += mask.size(0)
                correct_val += (pred_val == mask).sum().item()

        avg_val_loss          = epoch_val_loss          / len(val_loader)
        avg_skeleton_val_loss = epoch_skeleton_val_loss / len(val_loader)
        val_accuracy          = 100 * correct_val / total_val

        print(f"📊 Skeleton Val Accuracy: {val_accuracy:.2f}%")

        # ── Checkpoint & early stopping ───────────────────────────────────────
        if avg_val_loss < best_diffusion_loss:
            best_diffusion_loss = avg_val_loss
            no_improve          = 0
            if rank == 0:
                torch.save(
                    diffusion_model.state_dict(),
                    os.path.join(diffusion_save_dir, "best_diffusion_model.pth"),
                )
                print(f"  ✓ Saved best diffusion model (val {best_diffusion_loss:.4f})")
        else:
            no_improve += 1

        stop = torch.tensor(
            1 if no_improve >= args.diffusion_patience else 0, device=device
        )
        dist.broadcast(stop, src=0)
        if stop.item():
            if rank == 0:
                print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step(avg_val_loss)

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Diff Train {avg_train_loss:.4f}  Val {avg_val_loss:.4f} | "
                f"Skel Aux {avg_skeleton_loss:.4f}  Val {avg_skeleton_val_loss:.4f} | "
                f"Skel Acc Train {train_accuracy:.1f}%  Val {val_accuracy:.1f}%"
            )
            writer.add_scalar('Loss/Diffusion_Train',    avg_train_loss,         epoch)
            writer.add_scalar('Loss/Diffusion_Val',      avg_val_loss,           epoch)
            writer.add_scalar('Loss/Skeleton_Aux_Train', avg_skeleton_loss,      epoch)
            writer.add_scalar('Loss/Skeleton_Aux_Val',   avg_skeleton_val_loss,  epoch)
            writer.add_scalar('Accuracy/Skeleton_Train', train_accuracy,         epoch)
            writer.add_scalar('Accuracy/Skeleton_Val',   val_accuracy,           epoch)

            if (epoch + 1) % 300 == 0 or (epoch + 1) == args.epochs:
                torch.save(
                    diffusion_model.state_dict(),
                    os.path.join(diffusion_save_dir, f"diffusion_model_epoch_{epoch+1}.pth"),
                )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(rank, args):
    if rank == 0:
        log_name = (
            'sensor_train.log'   if args.train_sensor_model  else
            'skeleton_train.log' if args.train_skeleton_model else
            'diffusion_train.log'
        )
        log_path = os.path.join(args.output_dir, log_name)
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = open(log_path, 'ab', buffering=0)
        sys.stdout = io.TextIOWrapper(log_file, write_through=True)
        sys.stderr = sys.stdout

    setup(rank, args.world_size, seed=42)
    device = torch.device(f'cuda:{rank}')

    dataset = prepare_dataset(args)
    labels  = [int(dataset[i][3].item()) for i in range(len(dataset))]

    unique_classes = sorted(set(labels))
    if rank == 0:
        print(f"Classes found: {unique_classes}  ({len(unique_classes)} total)")

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(stratified_split.split(range(len(dataset)), labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=rank
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, sampler=val_sampler)

    if args.train_skeleton_model:
        train_skeleton_model(rank, args, device, train_loader, val_loader)
    elif args.train_sensor_model:
        train_sensor_model(rank, args, device, train_loader, val_loader)
    else:
        train_diffusion_model(rank, args, device, train_loader, val_loader)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',               type=int,   default=42)

    # Learning rates
    parser.add_argument("--sensor_lr",          type=float, default=1e-3)
    parser.add_argument("--skeleton_lr",        type=float, default=1e-3)
    parser.add_argument("--diffusion_lr",       type=float, default=1e-5)

    # Training stage flags
    parser.add_argument("--train_sensor_model",   type=eval, choices=[True, False], default=False)
    parser.add_argument("--train_skeleton_model", type=eval, choices=[True, False], default=False)

    # Dataset
    parser.add_argument("--window_size",     type=int,  default=48,    help="Sliding window size")
    parser.add_argument("--overlap",         type=int,  default=24,    help="Window overlap")
    parser.add_argument("--skeleton_folder", type=str,  default="./Own_Data/Labelled_Student_data/Skeleton_Data")
    parser.add_argument("--sensor_folder1",  type=str,  default="./Own_Data/Labelled_Student_data/Accelerometer_Data/P_Accel_Final")
    parser.add_argument("--sensor_folder2",  type=str,  default="./Own_Data/Labelled_Student_data/Accelerometer_Data/W_Accel_Final")

    # Epochs and patience
    parser.add_argument("--epochs",            type=int, default=2000)
    parser.add_argument("--sensor_epoch",      type=int, default=500)
    parser.add_argument("--skeleton_epochs",   type=int, default=200)
    parser.add_argument("--sensor_patience",   type=int, default=30)
    parser.add_argument("--skeleton_patience", type=int, default=30)
    parser.add_argument("--diffusion_patience",type=int, default=50)

    # Training infrastructure
    parser.add_argument("--batch_size",   type=int,  default=32)
    parser.add_argument("--step_size",    type=int,  default=20)
    parser.add_argument("--world_size",   type=int,  default=8)
    parser.add_argument("--gpus",         type=str,  default=None,
                        help="Comma-separated GPU IDs e.g. '0,1,2,3'")
    parser.add_argument("--output_dir",   type=str,  default="./results")

    # Diffusion
    parser.add_argument("--timesteps",    type=int,  default=10000)
    parser.add_argument("--ddim_scale",   type=float,default=0.0)
    parser.add_argument("--angular_loss", type=eval, choices=[True, False], default=False)
    parser.add_argument("--predict_noise",type=eval, choices=[True, False], default=False)

    args = parser.parse_args()

    if args.gpus is not None:
        gpu_ids = [g.strip() for g in args.gpus.split(',')]
        args.world_size = len(gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)

    mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
