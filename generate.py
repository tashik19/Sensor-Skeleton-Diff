"""
generate.py — Generate skeleton from sensor data

Usage 1: Pick your own sensor CSV files (recommended for evaluation)
    python generate.py \
        --phone_file  ./Cleaned/Phone/S29A10T01.csv \
        --watch_file  ./Cleaned/Watch/S29A10T01.csv \
        --activity    10

Usage 2: Random batch from dataset (shows generated vs real side-by-side)
    python generate.py

12-class activity labels:
    ADL:  1=DrinkWater 2=PickUp 3=Jacket 5=Sweep 6=Wash 7=Wave 8=Walk 9=SitStand
    Fall: 10=Backfall 11=Frontfall 13=Rightfall 14=Rotatefall
"""

import os, torch, argparse, imageio, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusion_model.model_loader import (
    load_sensor_model, load_diffusion_model_for_testing, NUM_CLASSES
)
from diffusion_model.skeleton_model import SkeletonTransformer
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from torch.utils.data import DataLoader
from diffusion_model.util import prepare_dataset
from diffusion_model.dataset import ALL_ACTIVITY_CODES

ACTIVITY_NAMES = {
    "1":"Drinking Water","2":"Pick Up Object","3":"Put On Jacket",
    "5":"Sweeping Floor","6":"Washing Hand",  "7":"Waving Hand",
    "8":"TUG/Walking",   "9":"Sit/Stand",
    "10":"Backfall","11":"Frontfall","13":"Rightfall","14":"Rotatefall",
}

# Correct connections for joint_order=[0,1,3,5,6,7,12,13,14,18,19,20,21,22,23,24]
# Position → AK joint: 0=Pelvis 1=SpineNavel 2=Neck
#   3=ShoulderL 4=ElbowL 5=WristL  6=ShoulderR 7=ElbowR 8=WristR
#   9=HipL 10=KneeL 11=AnkleL 12=FootL  13=HipR 14=KneeR 15=AnkleR
CONNECTIONS = [
    (0,1),(1,2),          # Pelvis→SpineNavel→Neck
    (2,3),(3,4),(4,5),    # Neck→ShoulderL→ElbowL→WristL
    (2,6),(6,7),(7,8),    # Neck→ShoulderR→ElbowR→WristR
    (0,9),(9,10),(10,11),(11,12),   # Pelvis→HipL→KneeL→AnkleL→FootL
    (0,13),(13,14),(14,15),         # Pelvis→HipR→KneeR→AnkleR
]


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_global_stats(phone_dir, watch_dir):
    all_vals = []
    for folder in [phone_dir, watch_dir]:
        for fname in os.listdir(folder):
            if not fname.endswith(".csv"): continue
            try:
                df = pd.read_csv(os.path.join(folder, fname), header=None)
                all_vals.append(df.values.astype(np.float32))
            except Exception: pass
    combined = np.concatenate(all_vals, axis=0)
    return combined.mean(axis=0).astype(np.float32), \
           (combined.std(axis=0)+1e-6).astype(np.float32)


def load_sensor_csv(path, window_size, sensor_mean, sensor_std):
    df   = pd.read_csv(path, header=None, names=["x","y","z"])
    data = df[["x","y","z"]].values.astype(np.float32)
    if len(data) >= window_size:
        data = data[-window_size:]
    else:
        pad  = np.repeat(data[-1:], window_size-len(data), axis=0)
        data = np.vstack([data, pad])
    data = (data - sensor_mean) / sensor_std
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # [1,T,3]


def draw_frame(ax, joints_xyz, title=""):
    ax.cla()
    ax.set_facecolor("#1a1a2e"); ax.grid(False); ax.set_axis_off()
    arm_joints = {2,3,4,5,6,7,8}
    leg_joints = {0,9,10,11,12,13,14,15}
    for j1,j2 in CONNECTIONS:
        x=[joints_xyz[j1,0],joints_xyz[j2,0]]
        y=[joints_xyz[j1,1],joints_xyz[j2,1]]
        z=[joints_xyz[j1,2],joints_xyz[j2,2]]
        if j1 in arm_joints or j2 in arm_joints: col="#4caf50"
        elif j1 in leg_joints or j2 in leg_joints: col="#f44336"
        else: col="#2196f3"
        ax.plot(x,y,z,"-",color=col,lw=2,alpha=0.9)
    ax.scatter(joints_xyz[:,0],joints_xyz[:,1],joints_xyz[:,2],
               color="white",s=20,zorder=5)
    ax.set_box_aspect([1,1,1])
    # Front-facing view: Azure Kinect Y=vertical(neg=up), Z=depth
    ax.view_init(elev=5, azim=180)
    ax.set_title(title, fontsize=8, color="white", pad=3)


def visualize_skeleton(generated, save_path, act_name="", real=None):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    T = generated.shape[1]; frames = []
    for fi in range(T):
        gen_j = generated[0,fi].reshape(16,3)
        if real is not None:
            fig = plt.figure(figsize=(12,6), facecolor="#0f0f1a")
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")
            draw_frame(ax1, gen_j, f"Generated — {act_name} (frame {fi+1}/{T})")
            draw_frame(ax2, real[0,fi].reshape(16,3), f"Real — {act_name}")
        else:
            fig = plt.figure(figsize=(6,6), facecolor="#0f0f1a")
            ax1 = fig.add_subplot(111, projection="3d")
            draw_frame(ax1, gen_j, f"Generated — {act_name} (frame {fi+1}/{T})")
        plt.tight_layout(pad=0.5); fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[:,:,:3].copy()); plt.close(fig)
    imageio.mimsave(save_path, frames, duration=0.15)
    print(f"GIF saved: {save_path}  ({T} frames)")


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Computing sensor stats...")
    sensor_mean, sensor_std = compute_global_stats(
        args.sensor_folder1, args.sensor_folder2
    )
    print(f"  mean={sensor_mean.round(3)}  std={sensor_std.round(3)}")

    sensor_model    = load_sensor_model(args, device)
    diffusion_model = load_diffusion_model_for_testing(
        device, args.output_dir, args.test_diffusion_model,
        skeleton_dim=args.window_size, num_classes=NUM_CLASSES,
    )
    skeleton_model = SkeletonTransformer(input_size=48, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.skeleton_model_path, map_location="cpu")
    ckpt = {k.replace("module.",""):v for k,v in ckpt.items()}
    skeleton_model.load_state_dict(ckpt, strict=False)
    skeleton_model.eval()

    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type="cosine", T=args.timesteps, step=1, device=device),
        device=device, ddim_scale=args.ddim_scale,
    )

    os.makedirs("./gif_tl", exist_ok=True)
    LABEL_TO_CODE = {v:k for k,v in ALL_ACTIVITY_CODES.items()}

    sensor_model.eval(); diffusion_model.eval()

    if args.phone_file and args.watch_file:
        # ── Mode 1: user-provided sensor files ───────────────────────────
        act_str = str(args.activity)
        if act_str not in ALL_ACTIVITY_CODES:
            raise ValueError(f"Activity {args.activity} not valid. "
                             f"Use: {sorted(ALL_ACTIVITY_CODES.keys())}")
        label_idx = ALL_ACTIVITY_CODES[act_str]
        act_name  = ACTIVITY_NAMES.get(act_str, f"Activity {args.activity}")
        print(f"Phone: {os.path.basename(args.phone_file)}")
        print(f"Watch: {os.path.basename(args.watch_file)}")
        print(f"Activity: {act_name} (label {label_idx})")

        s1 = load_sensor_csv(args.phone_file, args.window_size,
                             sensor_mean, sensor_std).to(device)
        s2 = load_sensor_csv(args.watch_file, args.window_size,
                             sensor_mean, sensor_std).to(device)
        label = torch.tensor([label_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            _, context = sensor_model(s1, s2, return_attn_output=True)
            if torch.isnan(context).any():
                print("NaN in context — check sensor files"); return
            generated = diffusion_process.generate(
                model=diffusion_model, context=context, label=label,
                shape=(1, args.window_size, 48),
                steps=args.timesteps, predict_noise=args.predict_noise,
            )
            gen_np = generated.cpu().numpy()
            pred   = skeleton_model(generated)
            pred_l = pred.argmax(dim=1).item()
            conf   = torch.softmax(pred,dim=1).max().item()
            pred_n = ACTIVITY_NAMES.get(str(LABEL_TO_CODE.get(pred_l,"?")),
                                        f"Label {pred_l}")
        print(f"Classifier prediction: {pred_n}  ({conf*100:.1f}% confidence)")

        save_path = f"./gif_tl/generated_{act_name.replace(' ','_')}.gif"
        visualize_skeleton(gen_np, save_path, act_name)

    else:
        # ── Mode 2: random batch, side-by-side with real skeleton ─────────
        dataset    = prepare_dataset(args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True)
        with torch.no_grad():
            skel_real, s1, s2, label = next(iter(dataloader))
            label_idx = label[0].item()
            act_code  = LABEL_TO_CODE.get(label_idx,"?")
            act_name  = ACTIVITY_NAMES.get(str(act_code), f"Activity {act_code}")
            print(f"Random batch activity: {act_name} (label {label_idx})")
            s1,s2,label = s1.to(device), s2.to(device), label.to(device)
            _, context  = sensor_model(s1, s2, return_attn_output=True)
            generated   = diffusion_process.generate(
                model=diffusion_model, context=context, label=label,
                shape=(args.batch_size, args.window_size, 48),
                steps=args.timesteps, predict_noise=args.predict_noise,
            )
        gen_np  = generated.cpu().numpy()
        real_np = skel_real.numpy()
        save_path = f"./gif_tl/generated_vs_real_{act_name.replace(' ','_')}.gif"
        visualize_skeleton(gen_np, save_path, act_name, real=real_np)


if __name__ == "__main__":
    p = argparse.ArgumentParser(epilog=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed",               type=int,   default=0)
    # Mode 1
    p.add_argument("--phone_file",         type=str,   default=None)
    p.add_argument("--watch_file",         type=str,   default=None)
    p.add_argument("--activity",           type=int,   default=10,
                   help="Activity code: 1-3,5-11,13,14")
    # Mode 2 / shared
    p.add_argument("--sensor_folder1",     type=str,   default="./Cleaned/Phone")
    p.add_argument("--sensor_folder2",     type=str,   default="./Cleaned/Watch")
    p.add_argument("--skeleton_folder",    type=str,   default="./Cleaned/Skeleton")
    p.add_argument("--overlap",            type=int,   default=16)
    p.add_argument("--batch_size",         type=int,   default=4)
    # Model
    p.add_argument("--output_dir",         type=str,   default="./results")
    p.add_argument("--skeleton_model_path",type=str,
                   default="./results/skeleton_model/best_skeleton_model.pth")
    p.add_argument("--test_diffusion_model",type=eval, choices=[True,False],
                   default=True)
    p.add_argument("--train_sensor_model", type=eval,  choices=[True,False],
                   default=False)
    # Diffusion
    p.add_argument("--window_size",        type=int,   default=32)
    p.add_argument("--timesteps",          type=int,   default=1000)
    p.add_argument("--ddim_scale",         type=float, default=0.0)
    p.add_argument("--predict_noise",      type=eval,  choices=[True,False],
                   default=False)
    main(p.parse_args())
