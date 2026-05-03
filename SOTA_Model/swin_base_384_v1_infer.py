# train_swin_base_384_v1_infer.py
# ✅ Swin Base 384 맞춤형 + Train + Inference(TTA rot90) + Fold Ensemble + Logits 저장 + Submission 생성
# - input: 384 fixed
# - model: prefer swin_base_patch4_window12_384.ms_in22k_ft_in1k (fallback to swin_base_patch4_window12_384)
# - train: grad accumulation + mixup(prob=0.5) + low lr + grad clip 1.0
# - inference: rot90 TTA(4) + fold ensemble(mean logits)
# - output: /root/outputs/swin_base_384_v1_infer/{best_swin_fold*.pth, predict_logits.pt, submission.csv}

import os
import gc
import random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


# ==========================================================================================
# 1. CONFIGURATION (Swin Base 384)
# ==========================================================================================
class Config:
    SEED = 42
    BASE_SIZE = 384

    #    384 전용 Swin-Base는 보통 window12_384 사용
    #    timm 환경에 따라 ms_in22k_ft_in1k가 없을 수 있어 fallback 로직으로 처리
    model_name_prefer = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
    model_name_fallback = "swin_base_patch4_window12_384"
    n_class = 17

    epochs = 40
    patience = 7

   
    batch_size = 4
    accumulation_steps = 4   # effective batch = 16
    val_batch_size = 2

    lr = 5e-5
    min_lr = 1e-6
    T_0 = 10

    use_mixup = True
    mixup_alpha = 0.2
    mixup_prob = 0.5
    grad_clip_max_norm = 1.0

    n_fold = 5
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PATHS
    DATA_ROOT = "/root/data2"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR  = os.path.join(DATA_ROOT, "test")
    TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
    TEST_CSV  = os.path.join(DATA_ROOT, "sample_submission.csv")

    # OUTPUT
    OUTPUT_DIR = "/root/outputs/swin_base_384_v1_infer"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TTA
    use_tta_rot90 = True


# ==========================================================================================
# 2. UTILS
# ==========================================================================================
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def correct_labels(csv_path: str):
    """strong baseline과 동일한 label correction"""
    print(f"\n[Label Correction] Processing {csv_path}...")
    df = pd.read_csv(csv_path)

    corrections = {
        "45f0d2dfc7e47c03.jpg": 7, "aec62dced7af97cd.jpg": 14, "0583254a73b48ece.jpg": 10,
        "1ec14a14bbe633db.jpg": 7, "c5182ab809478f12.jpg": 14, "8646f2c3280a4f49.jpg": 3,
        "38d1796b6ad99ddd.jpg": 10,
    }

    changed = 0
    col_name = "ID" if "ID" in df.columns else ("img_path" if "img_path" in df.columns else df.columns[0])

    for img_id, new_target in corrections.items():
        mask = df[col_name] == img_id
        if mask.any():
            prev = int(df.loc[mask, "target"].iloc[0])
            if prev != new_target:
                df.loc[mask, "target"] = new_target
                changed += 1

    new_path = csv_path.replace(".csv", "_corrected.csv")
    df.to_csv(new_path, index=False)
    print(f"  => {changed} labels corrected. Saved to {new_path}")
    return new_path


# ==========================================================================================
# 3. DATASET & TRANSFORMS (384 fixed)
# ==========================================================================================
class CustomDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        self.col_name = "ID" if "ID" in self.df.columns else ("img_path" if "img_path" in self.df.columns else self.df.columns[0])
        self.ids = self.df[self.col_name].values

        if mode != "test":
            self.labels = self.df["target"].values

        self.root_dir = Config.TEST_DIR if mode == "test" else Config.TRAIN_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = os.path.join(self.root_dir, img_id)

        image = cv2.imread(path)
        if image is None:
            image = np.zeros((Config.BASE_SIZE, Config.BASE_SIZE, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if self.mode != "test":
            return image, int(self.labels[index])
        return image


def get_transforms(mode="train"):
    #  지금은 Resize 384x384 고정. 
    if mode == "train":
        return A.Compose([
            A.Resize(Config.BASE_SIZE, Config.BASE_SIZE),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(max_holes=1, max_height=48, max_width=48, p=0.3),  # 384라 dropout 크기 소폭 증가
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(Config.BASE_SIZE, Config.BASE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ==========================================================================================
# 4. MODEL (Prefer in22k -> fallback)
# ==========================================================================================
def build_model():
    # timm 모델명 차이 대비: prefer → fallback 순으로 시도
    for name in [Config.model_name_prefer, Config.model_name_fallback]:
        try:
            model = timm.create_model(name, pretrained=True, num_classes=Config.n_class)
            print(f"Using model: {name}")
            return model
        except Exception as e:
            print(f"Failed to create model '{name}': {e}")
            continue
    raise RuntimeError("No valid Swin 384 model name found in this timm environment.")


# ==========================================================================================
# 5. TRAIN ONE FOLD (GRAD ACCUM)
# ==========================================================================================
def train_one_fold(fold, train_loader, val_loader, class_weights, mixup_fn):
    model = build_model().to(Config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0.05)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, eta_min=Config.min_lr)
    scaler = GradScaler()

    criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1.0
    patience_cnt = 0
    best_path = os.path.join(Config.OUTPUT_DIR, f"best_swin_fold{fold}.pth")

    for epoch in range(Config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0

        for i, (imgs, targets) in enumerate(tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch+1}/{Config.epochs}")):
            imgs = imgs.to(Config.device, non_blocking=True)
            targets = targets.to(Config.device, non_blocking=True)

            if mixup_fn is not None:
                imgs, targets = mixup_fn(imgs, targets)

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss = loss / Config.accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % Config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * Config.accumulation_steps

        scheduler.step()

        # Validation
        model.eval()
        preds, trues = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(Config.device, non_blocking=True)
                targets = targets.to(Config.device, non_blocking=True)

                out = model(imgs)
                loss = val_criterion(out, targets)
                val_loss += loss.item()

                preds.extend(out.argmax(1).cpu().tolist())
                trues.extend(targets.cpu().tolist())

        f1 = f1_score(trues, preds, average="macro") if len(preds) > 0 else 0.0
        print(f"[Fold {fold}] TrainLoss={train_loss/max(1,len(train_loader)):.4f} "
              f"| ValLoss={val_loss/max(1,len(val_loader)):.4f} | ValF1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ saved best: {best_path} (F1={best_f1:.4f})")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= Config.patience:
                print("  ⏹ Early Stopping Triggered.")
                break

    del model
    cleanup_cuda()


# ==========================================================================================
# 6. INFERENCE (TTA + FOLD ENSEMBLE) -> predict_logits.pt + submission.csv
# ==========================================================================================
@torch.no_grad()
def inference():
    print("\n✅ Starting Inference with TTA(rot90) & Fold Ensemble...")

    test_df = pd.read_csv(Config.TEST_CSV)
    test_ds = CustomDataset(test_df, transforms=get_transforms("valid"), mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    fold_logits_list = []

    for fold in range(Config.n_fold):
        ckpt = os.path.join(Config.OUTPUT_DIR, f"best_swin_fold{fold}.pth")
        if not os.path.exists(ckpt):
            print(f"  - Fold {fold}: checkpoint not found, skip -> {ckpt}")
            continue

        model = build_model().to(Config.device)
        model.load_state_dict(torch.load(ckpt, map_location=Config.device))
        model.eval()

        all_logits = []
        for imgs in tqdm(test_loader, desc=f"[Fold {fold}] Inference"):
            imgs = imgs.to(Config.device, non_blocking=True)

            if Config.use_tta_rot90:
                out = model(imgs).float()
                out += model(torch.rot90(imgs, 1, [2, 3])).float()
                out += model(torch.rot90(imgs, 2, [2, 3])).float()
                out += model(torch.rot90(imgs, 3, [2, 3])).float()
                out = out / 4.0
            else:
                out = model(imgs).float()

            all_logits.append(out.cpu())

        fold_logits = torch.cat(all_logits, dim=0)  # [N, C]
        fold_logits_list.append(fold_logits)

        del model
        cleanup_cuda()

    if len(fold_logits_list) == 0:
        raise RuntimeError("No fold checkpoints found. Inference aborted.")

    final_logits = torch.mean(torch.stack(fold_logits_list, dim=0), dim=0)  # [N, C]

    logits_path = os.path.join(Config.OUTPUT_DIR, "predict_logits.pt")
    torch.save(final_logits, logits_path)
    print("Logits saved to:", logits_path)

    final_probs = F.softmax(final_logits, dim=1).numpy()
    final_preds = np.argmax(final_probs, axis=1)

    sub = pd.read_csv(Config.TEST_CSV)
    sub["target"] = final_preds
    out_csv = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_csv, index=False)
    print("Submission saved to:", out_csv)


# ==========================================================================================
# 7. MAIN
# ==========================================================================================
def main():
    seed_everything(Config.SEED)

    assert os.path.isdir(Config.TRAIN_DIR), f"Missing TRAIN_DIR: {Config.TRAIN_DIR}"
    assert os.path.isdir(Config.TEST_DIR),  f"Missing TEST_DIR: {Config.TEST_DIR}"
    assert os.path.isfile(Config.TRAIN_CSV), f"Missing TRAIN_CSV: {Config.TRAIN_CSV}"
    assert os.path.isfile(Config.TEST_CSV), f"Missing sample_submission.csv: {Config.TEST_CSV}"

    train_csv = correct_labels(Config.TRAIN_CSV)
    df = pd.read_csv(train_csv)

    class_weights = compute_class_weight("balanced", classes=np.unique(df["target"]), y=df["target"])
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(Config.device)

    skf = StratifiedKFold(n_splits=Config.n_fold, shuffle=True, random_state=Config.SEED)

    mixup_fn = Mixup(
        mixup_alpha=Config.mixup_alpha,
        prob=Config.mixup_prob,
        mode="batch",
        label_smoothing=0.1,
        num_classes=Config.n_class,
    ) if Config.use_mixup else None

    for fold, (t_idx, v_idx) in enumerate(skf.split(df, df["target"])):
        print(f"\n==================== Fold {fold}/{Config.n_fold-1} ====================")

        train_ds = CustomDataset(df.iloc[t_idx], transforms=get_transforms("train"), mode="train")
        val_ds   = CustomDataset(df.iloc[v_idx], transforms=get_transforms("valid"), mode="valid")

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.val_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        train_one_fold(fold, train_loader, val_loader, class_weights, mixup_fn)

        del train_ds, val_ds, train_loader, val_loader
        cleanup_cuda()

    print("\n Training finished for all folds.")
    inference()


if __name__ == "__main__":
    # 메모리 파편화가 심하면 아래 옵션을 주석해제하고 커널/프로세스 재시작 후 실행
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    main()
