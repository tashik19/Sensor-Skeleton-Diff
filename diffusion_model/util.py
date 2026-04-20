import os
import torch
import imageio
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from .dataset import SlidingWindowDataset, read_csv_files, NUM_CLASSES


def _unwrap(model):
    """Return raw module, unwrapping DDP/DataParallel if needed."""
    return model.module if hasattr(model, "module") else model


def compute_loss(
    args, model, x0, label, context, t,
    mask=None, noise=None, device="cpu",
    diffusion_process=None, angular_loss=False,
    epoch=None, rank=0, batch_idx=0,
):
    """
    Diffusion loss (MSE on x0 prediction or noise prediction).
    Lip-reg removed — adds training cost without benefit at this stage.
    """
    x0      = x0.to(device)
    label   = label.to(device)
    context = context.to(device)
    t       = t.to(device)

    xt, true_noise = diffusion_process.add_noise(x0, t)
    xt         = xt.to(device)
    true_noise = true_noise.to(device)

    # ── Model prediction ─────────────────────────────────────────────────────
    predict_noise = bool(getattr(args, "predict_noise", False))

    if predict_noise:
        # DDPM mode: model predicts noise
        pred_noise = model(xt, context, t, sensor_pred=label).to(device)
        mse_loss   = F.mse_loss(pred_noise, true_noise)
        sqrt_ab    = diffusion_process.scheduler.sample_sqrt_a_bar_t(t).to(device)
        sqrt_1mab  = diffusion_process.scheduler.sample_sqrt_1_minus_a_bar_t(t).to(device)
        x0_pred    = (xt - sqrt_1mab * pred_noise) / (sqrt_ab + 1e-8)
    else:
        # x0-prediction mode (default, matches SSDL)
        x0_pred  = model(xt, context, t, sensor_pred=label).to(device)
        mse_loss = F.mse_loss(x0_pred, x0)

        # # Periodic skeleton GIF sanity check
        # if epoch is not None and batch_idx == 0 and rank == 0:
        #     if epoch in [99, 299, 499]:
        #         visualize_skeleton(
        #             x0[0].unsqueeze(0).cpu().detach().numpy(),
        #             save_path=f"./gif_tl/original_skeleton_{epoch}.gif",
        #         )
        #         visualize_skeleton(
        #             x0_pred[0].unsqueeze(0).cpu().detach().numpy(),
        #             save_path=f"./gif_tl/predicted_skeleton_{epoch}.gif",
        #         )

    total_loss = mse_loss

    # ── Optional angular loss ─────────────────────────────────────────────────
    if angular_loss:
        joint_angles      = compute_joint_angles(x0)
        pred_joint_angles = compute_joint_angles(x0_pred)
        angular_loss_val  = torch.norm(joint_angles - pred_joint_angles, p="fro")
        total_loss        = total_loss + 0.05 * angular_loss_val

    return total_loss, x0_pred


def add_random_noise(context, noise_std=0.01, noise_fraction=0.2):
    n   = context.size(0)
    k   = max(1, int(noise_fraction * n))
    idx = torch.randperm(n)[:k]
    context[idx] += torch.randn_like(context[idx]) * noise_std
    return context


def prepare_dataset(args):
    skeleton_data = read_csv_files(args.skeleton_folder)
    sensor_data1  = read_csv_files(args.sensor_folder1)
    sensor_data2  = read_csv_files(args.sensor_folder2)

    common_files = list(
        set(skeleton_data.keys()).intersection(sensor_data1.keys(), sensor_data2.keys())
    )
    if not common_files:
        raise ValueError("No common files found across skeleton/sensor1/sensor2.")

    for f in common_files:
        if skeleton_data[f].shape[1] == 97:
            skeleton_data[f] = skeleton_data[f].iloc[:, 1:]

    activity_codes = sorted(
        set(f.split("A")[1][:2].lstrip("0") for f in common_files)
    )
    label_encoder = OneHotEncoder(sparse_output=False)
    label_encoder.fit([[c] for c in activity_codes])
    print(f"Activity codes found: {activity_codes}  ({len(activity_codes)} classes)")

    dataset = SlidingWindowDataset(
        skeleton_data=skeleton_data,
        sensor1_data=sensor_data1,
        sensor2_data=sensor_data2,
        common_files=common_files,
        window_size=args.window_size,
        overlap=args.overlap,
        label_encoder=label_encoder,
        scaling="minmax",
    )
    return dataset


def compute_joint_angles(positions):
    joint_pairs = torch.tensor(
        [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
        device=positions.device,
    )
    B, T, F = positions.shape
    N   = F // 3
    pos = positions[:, :, :N * 3].view(B, T, N, 3)
    angles = []
    for i in range(0, T, 100):
        chunk  = pos[:, i:i + 100]
        v1     = chunk[:, :, joint_pairs[:, 1]] - chunk[:, :, joint_pairs[:, 0]]
        v2     = chunk[:, :, joint_pairs[:, 1]] - chunk[:, :, joint_pairs[:, 2]]
        denom  = torch.clamp(torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1), min=1e-6)
        cos_a  = torch.clamp(torch.sum(v1 * v2, dim=-1) / denom, -1 + 1e-7, 1 - 1e-7)
        angles.append(torch.acos(cos_a))
    return torch.cat(angles, dim=1)


def calculate_fid(real_activations, generated_activations):
    real = np.concatenate(real_activations, axis=0).reshape(len(real_activations[0]), -1)
    gen  = np.concatenate(generated_activations, axis=0).reshape(len(generated_activations[0]), -1)
    mu_r, sig_r = np.mean(real, axis=0), np.cov(real, rowvar=False)
    mu_g, sig_g = np.mean(gen,  axis=0), np.cov(gen,  rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sig_r @ sig_g, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm(sig_r @ sig_g + np.eye(sig_r.shape[0]) * 1e-6)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sig_r) + np.trace(sig_g) - 2 * np.trace(covmean)


def visualize_skeleton(positions, save_path="skeleton_animation.gif"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9),
        (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15),
    ]
    frames = []
    T = positions.shape[1]
    for fi in range(T):
        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white"); ax.grid(False); ax.set_axis_off()
        for j1, j2 in connections:
            c1 = positions[0, fi, j1 * 3:j1 * 3 + 3]
            c2 = positions[0, fi, j2 * 3:j2 * 3 + 3]
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], "o-", color="darkblue")
            ax.scatter(*c1, color="red", s=50)
            ax.scatter(*c2, color="red", s=50)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=-90, azim=-90)
        plt.tight_layout(); fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        frames.append(img.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
        plt.close(fig)
    imageio.mimsave(save_path, frames, duration=0.2)
    print(f"GIF saved: {save_path}")
