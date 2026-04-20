# Sensor-to-Skeleton Diffusion Model for Fall Detection

---

## Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib imageio tqdm tensorboard jupyter
```


---

## Dataset Setup

### 1. Download SmartFall MM

Download the raw SmartFall dataset and place the subject folders under:

```
data/
├── Phone/
│   ├── S01A01T01.csv
│   └── ...
├── Watch/
│   └── ...
└── Skeleton/
    └── ...
```

### 2. Run the cleaning notebook

The notebook reads raw data, removes corrupt files, trims recordings, and outputs a `Cleaned/` folder with three subfolders.

```bash
jupyter notebook dataset.ipynb
```

Run all cells. When finished, verify the output:

```
Cleaned/
├── Phone/        # ~919 phone accelerometer CSVs
├── Watch/        # ~919 watch accelerometer CSVs
└── Skeleton/     # ~919 skeleton CSVs (16-joint, 96 columns)
```

> **Note:** Activities A04 (Stepping Up) and A12 (Leftfall) are dropped automatically — zero valid files survive cleaning. The final dataset covers **12 classes**: activities 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14.

---

## Training

All three stages are run via `train.py`. Stages are selected by flags.

### Stage 1 — Sensor Classifier

```bash
python train.py \
  --train_sensor_model True \
  --window_size 32 \
  --overlap 16 \
  --skeleton_folder ./Cleaned/Skeleton \
  --sensor_folder1  ./Cleaned/Phone \
  --sensor_folder2  ./Cleaned/Watch \
  --sensor_epoch 500 \
  --sensor_patience 50 \
  --sensor_lr 1e-3 \
  --batch_size 32 \
  --output_dir ./results \
  --world_size 1
```

Expected output: `./results/sensor_model/best_sensor_model.pth`  
Target: val loss ≈ 0.45, val accuracy ≈ 81–82%

---

### Stage 2 — Skeleton Transformer

```bash
python train.py \
  --train_skeleton_model True \
  --window_size 32 \
  --overlap 16 \
  --skeleton_folder ./Cleaned/Skeleton \
  --sensor_folder1  ./Cleaned/Phone \
  --sensor_folder2  ./Cleaned/Watch \
  --skeleton_epochs 200 \
  --skeleton_lr 1e-3 \
  --step_size 20 \
  --batch_size 32 \
  --output_dir ./results \
  --world_size 1
```

Expected output: `./results/skeleton_model/best_skeleton_model.pth`  
Target: val loss ≈ 0.26, val accuracy ≈ 95–96%

---

### Stage 3 — Diffusion Model

> **Important:** Run Stage 1 and Stage 2 first. Stage 3 loads the frozen sensor model and frozen skeleton model from `./results/`.

```bash
python train.py \
  --window_size 32 \
  --overlap 16 \
  --skeleton_folder ./Cleaned/Skeleton \
  --sensor_folder1  ./Cleaned/Phone \
  --sensor_folder2  ./Cleaned/Watch \
  --epochs 2000 \
  --diffusion_patience 50 \
  --diffusion_lr 1e-5 \
  --batch_size 32 \
  --output_dir ./results \
  --world_size 1 \
  --timesteps 10000
```

Expected output: `./results/diffusion_model/best_diffusion_model.pth`  
Target: diffusion val loss ≈ 1.97, skeleton aux accuracy ≈ 97–99% throughout

Training log is written to `./results/diffusion_train.log`. Monitor with:

```bash
tail -f ./results/diffusion_train.log
```

---

## Multi-GPU Training

To use multiple GPUs, set `--world_size` to the number of GPUs and optionally specify which GPUs:

```bash
# Use 4 GPUs (IDs 0,1,2,3)
python train.py \
  --train_sensor_model True \
  --world_size 4 \
  --gpus 0,1,2,3 \
  [... other args ...]
```

---

## Generation

Generate a skeleton sequence from a pair of sensor files.

### Mode 1 — Your own sensor files

```bash
python generate.py \
  --phone_file ./Cleaned/Phone/S29A13T02.csv \
  --watch_file ./Cleaned/Watch/S29A13T02.csv \
  --activity 13
```

`--activity` is the **original activity code** (not the 0-indexed label):

| Code | Activity | Type |
|------|----------|------|
| 1 | Drink Water | ADL |
| 2 | Eat Meal | ADL |
| 3 | Walking | ADL |
| 5 | Sweeping Floor | ADL |
| 6 | Washing Hand | ADL |
| 7 | Waving Hand | ADL |
| 8 | TUG/Walking | ADL |
| 9 | Sit Down | ADL |
| 10 | Backfall | Fall |
| 11 | Frontfall | Fall |
| 13 | Rightfall | Fall |
| 14 | Rotatefall | Fall |

Output: GIF saved to `./gif_tl/generated_<ActivityName>.gif`  
Terminal: `Classifier prediction: <ActivityName>  (XX.X% confidence)`

### Mode 2 — Random batch from dataset

```bash
python generate.py \
  --window_size 32 \
  --skeleton_folder ./Cleaned/Skeleton \
  --sensor_folder1  ./Cleaned/Phone \
  --sensor_folder2  ./Cleaned/Watch \
  --output_dir ./results \
  --timesteps 1000 \
  --batch_size 4 \
  --test_diffusion_model True
```

---

## Results

Evaluated on unseen subjects (S48, S51) — no data from these subjects was seen during training.

| Activity | Type | Subject | Classifier Confidence |
|----------|------|---------|----------------------|
| Rightfall | Fall | S29 | 99.8% |
| Backfall | Fall | S51 | 99.7% |
| Sweeping Floor | ADL | S48 | 99.9% |
| Washing Hand | ADL | S48 | 100.0% |
| Waving Hand | ADL | S48 | 98.0% |
| TUG/Walking | ADL | S48 | 98.5% |

