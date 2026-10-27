"""
train.py
─────────────────────────────────────────────────────────────────────────────
Full training loop for the Violence Detection model.

Features
--------
  • Resumes from any checkpoint (epoch number + full state stored in .pt file)
  • Uploads model + metrics to Hugging Face Hub after EVERY epoch
  • Saves confusion matrix (as image + raw numpy), per-epoch metrics CSV,
    and a JSON log of every metric
  • Mixed-precision (AMP) training
  • Cosine-annealing LR schedule with warm-up
  • Weighted random sampler to handle class imbalance

Usage
-----
# First run
python train.py \
    --data_dir /path/to/preprocessed \
    --hf_repo   your-username/violence-detector \
    --hf_token  hf_xxxxxxxxxxxx \
    --epochs 30

# Resume from checkpoint
python train.py \
    --data_dir  /path/to/preprocessed \
    --hf_repo   your-username/violence-detector \
    --hf_token  hf_xxxxxxxxxxxx \
    --epochs 30 \
    --resume    checkpoints/checkpoint_epoch_07.pt
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import ViolenceDataset, ViolenceClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─── HuggingFace upload helper ────────────────────────────────────────────────

def upload_to_hf(
    local_dir: Path,
    repo_id: str,
    token: str,
    commit_message: str,
):
    """Upload the entire `local_dir` folder to a HF Hub model repo."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        token=token,
        commit_message=commit_message,
    )
    log.info("✅ Uploaded to https://huggingface.co/%s", repo_id)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    metrics: dict,
    best_val_f1: float,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "best_val_f1": best_val_f1,
        },
        path,
    )
    log.info("💾 Checkpoint saved → %s", path)
    return path


def load_checkpoint(path: str, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_f1 = ckpt.get("best_val_f1", 0.0)
    log.info("▶ Resuming from epoch %d (best val F1 so far: %.4f)", start_epoch, best_val_f1)
    return start_epoch, best_val_f1, ckpt.get("metrics", {})


# ─── Metrics helpers ─────────────────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, epoch: int, out_dir: Path):
    """Save confusion matrix as a PNG and as a raw .npy file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"confusion_matrix_epoch_{epoch:03d}.npy"
    np.save(npy_path, cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ["nonviolence", "violence"]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix – Epoch {epoch}",
    )
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    png_path = out_dir / f"confusion_matrix_epoch_{epoch:03d}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path, npy_path


# ─── Training / evaluation loops ─────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    device: str,
    train: bool,
) -> tuple[float, list, list]:
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for clips, labels in loader:
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=device != "cpu"):
                logits = model(clips)
                loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    return avg_loss, all_preds, all_labels


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Output directories ───────────────────────────────────────────────────
    out_dir   = Path(args.out_dir)
    ckpt_dir  = out_dir / "checkpoints"
    metrics_dir = out_dir / "metrics"
    best_dir  = out_dir / "best_model"
    for d in [ckpt_dir, metrics_dir, best_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_ds = ViolenceDataset(data_dir / "train.csv", train=True,  img_size=args.img_size)
    val_ds   = ViolenceDataset(data_dir / "val.csv",   train=False, img_size=args.img_size)

    # Weighted sampler to handle class imbalance
    labels_arr = np.array([int(r["label"]) for r in train_ds.records])
    class_counts = np.bincount(labels_arr)
    weights = 1.0 / class_counts[labels_arr]
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = ViolenceClassifier(num_classes=2, pretrained=not bool(args.resume)).to(device)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )
    scaler = GradScaler()

    # Class-weighted loss
    cw = torch.tensor(class_counts.max() / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_f1 = 0.0
    all_epoch_metrics: list[dict] = []

    if args.resume:
        start_epoch, best_val_f1, prev_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        if isinstance(prev_metrics, list):
            all_epoch_metrics = prev_metrics

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        log.info("═" * 60)
        log.info("Epoch %d / %d", epoch, args.epochs)

        # Train
        train_loss, train_preds, train_labels = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True
        )
        scheduler.step()

        # Validate
        val_loss, val_preds, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False
        )

        # Metrics
        val_f1  = f1_score(val_labels, val_preds, average="macro")
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except Exception:
            val_auc = float("nan")

        cm = confusion_matrix(val_labels, val_preds)
        report = classification_report(
            val_labels, val_preds,
            target_names=["nonviolence", "violence"], output_dict=True
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
            "val_auc": val_auc,
            "val_precision_violence": report["violence"]["precision"],
            "val_recall_violence": report["violence"]["recall"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": round(time.time() - t0, 1),
        }
        all_epoch_metrics.append(epoch_metrics)

        log.info(
            "Train loss: %.4f | Val loss: %.4f | Val Acc: %.4f | Val F1: %.4f | AUC: %.4f",
            train_loss, val_loss, val_acc, val_f1, val_auc,
        )

        # ── Save confusion matrix ─────────────────────────────────────────
        png_path, npy_path = save_confusion_matrix(cm, epoch, metrics_dir)
        log.info("Confusion matrix → %s", png_path)

        # ── Save epoch metrics CSV ────────────────────────────────────────
        metrics_csv = metrics_dir / "all_epochs_metrics.csv"
        import csv as _csv
        fieldnames = list(all_epoch_metrics[0].keys())
        with open(metrics_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_epoch_metrics)

        # ── Save epoch metrics JSON ───────────────────────────────────────
        metrics_json = metrics_dir / "all_epochs_metrics.json"
        with open(metrics_json, "w") as f:
            json.dump(all_epoch_metrics, f, indent=2)

        # ── Checkpoint (always) ───────────────────────────────────────────
        ckpt_path = save_checkpoint(
            ckpt_dir, epoch, model, optimizer, scheduler, scaler,
            all_epoch_metrics, best_val_f1,
        )

        # ── Best model ────────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_f1": val_f1},
                best_dir / "best_model.pt",
            )
            log.info("🏆 New best model saved (val F1 = %.4f)", val_f1)

        # ── Upload to Hugging Face ────────────────────────────────────────
        if args.hf_repo and args.hf_token:
            try:
                # Prepare upload dir: checkpoint + metrics
                hf_stage = out_dir / "_hf_upload"
                hf_stage.mkdir(exist_ok=True)

                import shutil
                shutil.copy(ckpt_path, hf_stage / ckpt_path.name)
                shutil.copy(metrics_csv, hf_stage / "metrics.csv")
                shutil.copy(metrics_json, hf_stage / "metrics.json")
                shutil.copy(png_path,    hf_stage / png_path.name)

                # Also copy best model if updated this epoch
                best_pt = best_dir / "best_model.pt"
                if best_pt.exists():
                    shutil.copy(best_pt, hf_stage / "best_model.pt")

                # Write a model card
                model_card = hf_stage / "README.md"
                model_card.write_text(
                    f"# Violence Detector – Video Swin Transformer\n\n"
                    f"**Last epoch:** {epoch} / {args.epochs}  \n"
                    f"**Best Val F1:** {best_val_f1:.4f}  \n"
                    f"**Val Accuracy:** {val_acc:.4f}  \n\n"
                    f"## Labels\n`0` = nonviolence, `1` = violence\n\n"
                    f"## Metrics (epoch {epoch})\n"
                    + "\n".join(f"- **{k}**: {v}" for k, v in epoch_metrics.items())
                )

                upload_to_hf(
                    hf_stage, args.hf_repo, args.hf_token,
                    commit_message=f"Epoch {epoch}: val_f1={val_f1:.4f} val_acc={val_acc:.4f}",
                )
            except Exception as e:
                log.warning("HF upload failed (epoch %d): %s", epoch, e)
        else:
            log.info("(HF upload skipped – no --hf_repo / --hf_token provided)")

    log.info("✅ Training complete. Best Val F1: %.4f", best_val_f1)


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True,  help="Preprocessed dataset directory")
    p.add_argument("--out_dir",     default="runs/exp1")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--resume",      default=None, help="Path to checkpoint .pt to resume from")
    p.add_argument("--hf_repo",     default=None, help="HuggingFace repo, e.g. user/violence-detector")
    p.add_argument("--hf_token",    default=None, help="HuggingFace write token")
    args = p.parse_args()
    train(args)
