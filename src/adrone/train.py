import argparse, json, torch, torch.nn as nn, torch.optim as optim, numpy as np, random
from torch.utils.data import DataLoader
from .data.dataset import MelDataset
from .models.cnn_small import CNNSmall
from tqdm import tqdm
import os, yaml

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    # Use labels path from config if specified, otherwise use default
    labels_json = cfg.get("labels_json", "data/processed/labels.json")
    train_ds = MelDataset(cfg["train_csv"], labels_json, cfg["sample_rate"], cfg["n_mels"], cfg["n_fft"], cfg["hop_length"], cfg["window_sec"])
    val_ds   = MelDataset(cfg["val_csv"],   labels_json, cfg["sample_rate"], cfg["n_mels"], cfg["n_fft"], cfg["hop_length"], cfg["window_sec"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    n_classes = len(train_ds.labels)
    model = CNNSmall(n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        # Clear cache at start
        torch.cuda.empty_cache()
    else:
        print("   ⚠️  Warning: Training on CPU will be much slower!")
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg["epochs"]):
        model.train(); loss_sum = 0
        # Show GPU memory usage
        if device == "cuda":
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n💾 GPU Memory - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
        
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            loss_sum += loss.item()

        # val
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct/total if total else 0
        print(f"val_acc={acc:.4f}  train_loss={loss_sum/len(train_dl):.4f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), cfg["model_out"])
            with open(cfg["labels_out"], "w") as f:
                json.dump({"labels": train_ds.labels}, f, indent=2)

    print(f"Best val acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
