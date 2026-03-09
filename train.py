import warnings
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TextPaletteDataset
from model import Text2PaletteModel
from config import DATA_DIR, PROCESSED_DIR, EMB_TRAIN, EMB_VAL, DEVICE, BATCH_SIZE, EPOCHS, LR, N_COLORS


# ── Loss functions ────────────────────────────────────────────────

def triplet_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Batch-hard triplet loss.
    Anchor = pred, positive = target, negative = target of next sample in batch.
    """
    B = pred.size(0)
    pred_flat   = pred.view(B, -1)
    target_flat = target.view(B, -1)
    neg_flat    = torch.roll(target_flat, shifts=1, dims=0)
    d_pos = F.pairwise_distance(pred_flat, target_flat)
    d_neg = F.pairwise_distance(pred_flat, neg_flat)
    return F.relu(d_pos - d_neg + margin).mean()


def diversity_penalty(palette: torch.Tensor, min_dist: float = 0.12) -> torch.Tensor:
    """Penalises colour pairs closer than min_dist in Oklab space."""
    total, count = 0.0, 0
    for i in range(N_COLORS):
        for j in range(i + 1, N_COLORS):
            dist   = torch.norm(palette[:, i, :] - palette[:, j, :], dim=-1)
            total += (F.relu(min_dist - dist) ** 2).mean()
            count += 1
    return total / count


def lightness_spread_loss(palette: torch.Tensor) -> torch.Tensor:
    """Encourages L values to span [0.1, 0.9] evenly across the 5 colours."""
    L_sorted, _ = torch.sort(palette[:, :, 0], dim=1)
    targets = torch.linspace(0.1, 0.9, N_COLORS, device=palette.device)
    return F.mse_loss(L_sorted, targets.unsqueeze(0).expand_as(L_sorted))


# ── Training loop ─────────────────────────────────────────────────

def main():
    train_ds = TextPaletteDataset(PROCESSED_DIR / "train.csv", EMB_TRAIN)
    val_ds   = TextPaletteDataset(PROCESSED_DIR / "val.csv",   EMB_VAL)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {DEVICE}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = Text2PaletteModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    best_val  = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_huber = tot_trip = tot_div = tot_spread = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", dynamic_ncols=True)
        for emb, target, w in pbar:
            emb  = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            pred = model(emb)

            # Weighted Huber: tag-based samples contribute 4× more
            huber_elem = F.huber_loss(pred, target, delta=0.1, reduction="none")
            huber      = (huber_elem.mean(dim=[1, 2]) * w).mean()
            triplet    = triplet_loss(pred, target)
            div_pen    = diversity_penalty(pred)
            spread     = lightness_spread_loss(pred)
            loss       = huber + 0.3 * triplet + 1.0 * div_pen + 0.5 * spread

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_huber  += huber.item()
            tot_trip   += triplet.item()
            tot_div    += div_pen.item()
            tot_spread += spread.item()
            pbar.set_postfix(huber=f"{huber.item():.4f}", trip=f"{triplet.item():.4f}", div=f"{div_pen.item():.4f}", spread=f"{spread.item():.4f}")

        scheduler.step()
        n = len(train_loader)

        model.eval()
        val_huber = 0.0
        with torch.no_grad():
            for emb, target, _ in val_loader:
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                val_huber += F.huber_loss(model(emb), target, delta=0.1).item()

        avg_val = val_huber / len(val_loader)
        print(f"Epoch {epoch:3d} | huber={tot_huber/n:.4f} trip={tot_trip/n:.4f} "
              f"div={tot_div/n:.4f} spread={tot_spread/n:.4f} | "
              f"val_huber={avg_val:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), DATA_DIR / "best_palette_gen.pt")
            print(f"  Saved (val_huber={avg_val:.4f})")

    print(f"Training complete. Best val_huber={best_val:.4f}")


if __name__ == "__main__":
    main()
