# train_convnext_base_single.py
# ✅ ConvNeXt Base 전용 "단일 실행" 스크립트
# - 기존 strong_baseline 파라미터/로직 그대로 유지
# - model_name만 convnext_base.fb_in22k_ft_in1k 로 변경
# - OOM 방지: (1) val batch만 줄이기 (2) pin_memory/worker 유지 (3) per-fold clean-up
# - 출력 폴더: /root/outputs/strong_baseline_convnext_base

import os
import gc
import random
import math
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
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
# 1. CONFIGURATION
# ==========================================================================================
class Config:
    SEED = 42
    BASE_SIZE = 640

    # ✅ ONLY CHANGE: model_name
    model_name = 'convnext_base.fb_in22k_ft_in1k'
    n_class = 17

    epochs = 50
    patience = 5
    batch_size = 16              # ✅ 기존 유지
    val_batch_size = 8           # ✅ OOM 방지: validation만 줄이기 
    lr = 1e-4
    min_lr = 1e-6
    T_0 = 10

    # Advanced Augmentation
    use_mixup = True
    mixup_alpha = 0.2
    mixup_prob = 1.0
    grad_clip_max_norm = 5.0

    n_fold = 5
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ✅ SERVER PATHS
    DATA_ROOT = "/root/data2"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR  = os.path.join(DATA_ROOT, "test")
    TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
    TEST_CSV  = os.path.join(DATA_ROOT, "sample_submission.csv")

    # ✅ OUTPUT
    OUTPUT_DIR = "/root/outputs/strong_baseline_convnext_base"
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================================================
# 2. UTILS & LABEL CORRECTION
# ==========================================================================================
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def correct_labels(csv_path: str):
    print(f"\n[Label Correction] Processing {csv_path}...")
    df = pd.read_csv(csv_path)

    corrections = {
        "45f0d2dfc7e47c03.jpg": 7, "aec62dced7af97cd.jpg": 14, "0583254a73b48ece.jpg": 10,
        "1ec14a14bbe633db.jpg": 7, "c5182ab809478f12.jpg": 14, "8646f2c3280a4f49.jpg": 3,
        "38d1796b6ad99ddd.jpg": 10,
    }

    changed = 0
    col_name = 'ID' if 'ID' in df.columns else ('img_path' if 'img_path' in df.columns else df.columns[0])

    for img_id, new_target in corrections.items():
        mask = df[col_name] == img_id
        if mask.any():
            prev = int(df.loc[mask, 'target'].iloc[0])
            if prev != new_target:
                df.loc[mask, 'target'] = new_target
                changed += 1

    new_path = csv_path.replace('.csv', '_corrected.csv')
    df.to_csv(new_path, index=False)
    print(f"  => {changed} labels corrected. Saved to {new_path}")
    return new_path


def cleanup_cuda():
    """✅ fold 끝날 때마다 메모리 정리 (중요)"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ==========================================================================================
# 3. ASPECT RATIO BUCKETING
# ==========================================================================================
def make_bucket_resolutions(base_size=640, min_dim=320, max_dim=1024, step=32):
    resolutions = set()
    target_area = base_size * base_size

    w = min_dim
    while w <= max_dim:
        h = max(1, int(target_area / w))
        w_snap = max(step, (w // step) * step)
        h_snap = max(step, (h // step) * step)

        if min_dim <= h_snap <= max_dim:
            resolutions.add((w_snap, h_snap))
            resolutions.add((h_snap, w_snap))
        w += step

    return sorted(list(resolutions))


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_resolutions, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_resolutions = bucket_resolutions
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {res: [] for res in bucket_resolutions}
        self._assign_buckets()

    def _assign_buckets(self):
        print("Grouping images into buckets...")
        for idx in range(len(self.dataset)):
            w, h = self.dataset.get_image_size(idx)
            ar = float(w) / float(h) if h != 0 else 1.0

            best_res = min(self.bucket_resolutions, key=lambda res: abs((res[0] / res[1]) - ar))
            self.buckets[best_res].append(idx)

            try:
                self.dataset.set_target_resolution(idx, best_res)
            except Exception:
                pass

    def __iter__(self):
        batches = []
        for _, indices in self.buckets.items():
            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        count = 0
        for indices in self.buckets.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += math.ceil(len(indices) / self.batch_size)
        return count


# ==========================================================================================
# 4. PREPROCESSING & AUGMENTATION (Deskew & Letterbox)
# ==========================================================================================
class Preprocessor:
    @staticmethod
    def deskew(image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            coords = np.column_stack(np.where(thresh > 0))
            if coords.size == 0:
                return image

            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if abs(angle) < 0.5:
                return image

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return image

    @staticmethod
    def letterbox(image, target_size):
        target_w, target_h = target_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        delta_w = target_w - nw
        delta_h = target_h - nh
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return new_image


def get_transforms(mode='train'):
    common = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    if mode == 'train':
        aug = [
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.CoarseDropout(max_holes=1, max_height=64, max_width=64, p=0.2),
        ]
        return A.Compose(aug + common)
    return A.Compose(common)


# ==========================================================================================
# 5. DATASET
# ==========================================================================================
class RepeatDataset(Dataset):
    def __init__(self, dataset, repeats=4):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

    def get_image_size(self, idx):
        return self.dataset.get_image_size(idx % len(self.dataset))

    def set_target_resolution(self, idx, res):
        self.dataset.set_target_resolution(idx % len(self.dataset), res)


class CustomDataset(Dataset):
    def __init__(self, df, transforms=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        self.col_name = 'ID' if 'ID' in df.columns else ('img_path' if 'img_path' in df.columns else df.columns[0])
        self.ids = df[self.col_name].values

        if mode != 'test':
            self.labels = df['target'].values

        self.image_sizes = []
        self.target_resolutions = {}

        self.root_dir = Config.TEST_DIR if mode == 'test' else Config.TRAIN_DIR

        print(f"Scanning image sizes for {mode}... root={self.root_dir}")
        for img_id in tqdm(self.ids):
            path = os.path.join(self.root_dir, img_id)
            try:
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    self.image_sizes.append((w, h))
                else:
                    self.image_sizes.append((Config.BASE_SIZE, Config.BASE_SIZE))
            except Exception:
                self.image_sizes.append((Config.BASE_SIZE, Config.BASE_SIZE))

    def __len__(self):
        return len(self.df)

    def get_image_size(self, idx):
        return self.image_sizes[idx]

    def set_target_resolution(self, idx, res):
        self.target_resolutions[idx] = res

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = os.path.join(self.root_dir, img_id)

        image = cv2.imread(path)
        if image is None:
            image = np.zeros((Config.BASE_SIZE, Config.BASE_SIZE, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Preprocessor.deskew(image)

        target_res = self.target_resolutions.get(index, (Config.BASE_SIZE, Config.BASE_SIZE))
        image = Preprocessor.letterbox(image, target_res)

        if self.transforms:
            image = self.transforms(image=image)['image']

        if self.mode != 'test':
            return image, int(self.labels[index])
        return image


# ==========================================================================================
# 6. MODEL
# ==========================================================================================
class Network(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.num_features = self.backbone.num_features
        self.head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


# ==========================================================================================
# 7. TRAINING LOOP
# ==========================================================================================
def main_train():
    seed_everything(Config.SEED)

    assert os.path.isdir(Config.TRAIN_DIR), f"Missing TRAIN_DIR: {Config.TRAIN_DIR}"
    assert os.path.isdir(Config.TEST_DIR),  f"Missing TEST_DIR:  {Config.TEST_DIR}"
    assert os.path.isfile(Config.TRAIN_CSV), f"Missing TRAIN_CSV: {Config.TRAIN_CSV}"
    assert os.path.isfile(Config.TEST_CSV), f"Missing sample_submission.csv: {Config.TEST_CSV}"

    train_csv = correct_labels(Config.TRAIN_CSV)
    df = pd.read_csv(train_csv)

    class_weights = compute_class_weight('balanced', classes=np.unique(df['target']), y=df['target'])
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(Config.device)

    skf = StratifiedKFold(n_splits=Config.n_fold, shuffle=True, random_state=Config.SEED)

    mixup_fn = Mixup(
        mixup_alpha=Config.mixup_alpha, prob=Config.mixup_prob, mode='batch',
        label_smoothing=0.1, num_classes=Config.n_class
    ) if Config.use_mixup else None

    bucket_res_list = make_bucket_resolutions(Config.BASE_SIZE)
    print(f"Generated {len(bucket_res_list)} buckets.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        print(f"\n[Fold {fold+1}/{Config.n_fold}] Training Start... ({Config.model_name})")

        train_sub = df.iloc[train_idx].reset_index(drop=True)
        val_sub   = df.iloc[val_idx].reset_index(drop=True)

        train_ds = CustomDataset(train_sub, transforms=get_transforms('train'), mode='train')
        train_ds_rep = RepeatDataset(train_ds, repeats=4)
        val_ds = CustomDataset(val_sub, transforms=get_transforms('valid'), mode='valid')

        train_sampler = BucketBatchSampler(train_ds_rep, Config.batch_size, bucket_res_list, shuffle=True)
        train_loader = DataLoader(
            train_ds_rep, batch_sampler=train_sampler,
            num_workers=Config.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=Config.val_batch_size, shuffle=False,
            num_workers=Config.num_workers, pin_memory=True
        )

        model = Network(Config.model_name, Config.n_class).to(Config.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, eta_min=Config.min_lr)
        scaler = GradScaler()

        criterion = SoftTargetCrossEntropy()
        val_criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_f1 = 0.0
        patience_counter = 0
        best_path = os.path.join(Config.OUTPUT_DIR, f"best_model_fold{fold}.pth")

        for epoch in range(Config.epochs):
            model.train()
            train_loss = 0.0

            for imgs, targets in tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch+1}/{Config.epochs}"):
                imgs = imgs.to(Config.device, non_blocking=True)
                targets = targets.to(Config.device, non_blocking=True)

                if mixup_fn is not None:
                    imgs, targets = mixup_fn(imgs, targets)

                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            preds, trues = [], []
            with torch.no_grad():
                for imgs, targets in tqdm(val_loader, desc=f"[Fold {fold}] Validating"):
                    imgs = imgs.to(Config.device, non_blocking=True)
                    targets = targets.to(Config.device, non_blocking=True)

                    outputs = model(imgs)
                    loss = val_criterion(outputs, targets)

                    val_loss += loss.item()
                    preds.extend(outputs.argmax(1).cpu().tolist())
                    trues.extend(targets.cpu().tolist())

            f1 = f1_score(trues, preds, average='macro') if len(preds) > 0 else 0.0
            print(f"[Fold {fold}] TrainLoss={train_loss/max(1,len(train_loader)):.4f} "
                  f"| ValLoss={val_loss/max(1,len(val_loader)):.4f} | ValF1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)
                print(f"   saved best: {best_path} (F1={best_f1:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.patience:
                    print("  ⏹ Early Stopping Triggered.")
                    break

        #  fold 끝나면 정리 (OOM 방지 핵심)
        del model, train_ds, train_ds_rep, val_ds, train_loader, val_loader, train_sampler
        cleanup_cuda()

    print("\n Training finished for all folds.")


# ==========================================================================================
# 8. INFERENCE (TTA & ENSEMBLE)
# ==========================================================================================
@torch.no_grad()
def inference():
    print("\nStarting Inference with TTA & Ensemble...")

    test_df = pd.read_csv(Config.TEST_CSV)
    test_ds = CustomDataset(test_df, transforms=get_transforms('valid'), mode='test')
    test_loader = DataLoader(
        test_ds, batch_size=Config.val_batch_size, shuffle=False,
        num_workers=Config.num_workers, pin_memory=True
    )

    fold_preds = []

    for fold in range(Config.n_fold):
        path = os.path.join(Config.OUTPUT_DIR, f"best_model_fold{fold}.pth")
        if not os.path.exists(path):
            print(f"Model for fold {fold} not found at {path}, skipping...")
            continue

        model = Network(Config.model_name, Config.n_class).to(Config.device)
        model.load_state_dict(torch.load(path, map_location=Config.device))
        model.eval()

        preds_batches = []
        for imgs in tqdm(test_loader, desc=f"Fold {fold} Inference"):
            imgs = imgs.to(Config.device, non_blocking=True)

            out = model(imgs).float()
            out += model(torch.rot90(imgs, 1, [2, 3])).float()
            out += model(torch.rot90(imgs, 2, [2, 3])).float()
            out += model(torch.rot90(imgs, 3, [2, 3])).float()

            preds_batches.append((out / 4.0).cpu().numpy())

        preds_all = np.vstack(preds_batches)
        fold_preds.append(preds_all)

        del model
        cleanup_cuda()

    if len(fold_preds) == 0:
        print("Error: No models found for inference.")
        return

    final_logits = np.mean(np.stack(fold_preds, axis=0), axis=0)

    logits_path = os.path.join(Config.OUTPUT_DIR, "predict_logits.pt")
    torch.save(torch.tensor(final_logits), logits_path)
    print("Logits saved to:", logits_path)

    final_probs = F.softmax(torch.tensor(final_logits), dim=1).numpy()
    final_preds = np.argmax(final_probs, axis=1)

    sub = pd.read_csv(Config.TEST_CSV)
    sub['target'] = final_preds
    out_csv = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_csv, index=False)
    print("Inference Done. Submission saved:", out_csv)


if __name__ == '__main__':
    # fragmentation 완화 옵션 
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    main_train()
    inference()
