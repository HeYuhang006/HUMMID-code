"""Train the two-stage HUMMID model on preprocessed HCP volumes."""

import argparse
import logging
from pathlib import Path

from scipy.io import savemat
import torch

from dataloader import Nii_dataloader, read_subjects
from model.factory import checkpoint_paths, load_model_safe, model_pair

from .losses import combined_mae_ssim_loss


def run_epoch(model, loader, device, optimizer=None, t1_model=None):
    """Use the HCP MAE–SSIM objective for either training stage."""
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for inputs, targets, masks in loader:
            inputs, targets, masks = (x.to(device) for x in (inputs, targets, masks))
            if training:
                optimizer.zero_grad()
            if t1_model is None:
                prediction = model(inputs, masks)[..., :-1]
                reference = targets[..., -2:-1]
            else:
                with torch.no_grad():
                    t1w = t1_model(inputs, masks)[..., :-1]
                dwi_inputs = torch.cat((inputs[..., 4:], t1w), dim=-1)
                prediction = model(dwi_inputs, masks)[..., :-1]
                reference = targets[..., :-2]
            loss = combined_mae_ssim_loss(prediction, reference, masks)
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def main(cnn=False, argv=None):
    parser = argparse.ArgumentParser(description=__doc__ if not cnn else "Train the CNN ablation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-subjects", type=Path, required=True)
    parser.add_argument("--val-subjects", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory for checkpoints and loss history.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-flip", action="store_true", help="Disable left-right flipping on both splits.")
    parser.add_argument("--init-t1w", type=Path, help="Optional initial T1w state dict.")
    parser.add_argument("--init-dwi", type=Path, help="Optional initial DWI state dict.")
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    train_names = read_subjects(args.train_subjects)
    val_names = read_subjects(args.val_subjects)
    if set(train_names) & set(val_names):
        parser.error("training and validation lists must be disjoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")

    paths = checkpoint_paths(args.model_dir, cnn)
    history_path = args.model_dir / ("loss_cnn.mat" if cnn else "loss_hummid.mat")
    if any(path.exists() for path in (*paths, history_path)):
        parser.error("choose a new model directory to keep the existing checkpoints")

    loader = Nii_dataloader(args.data_root, train_names, val_names, flip=not args.no_flip)
    train_loader, val_loader = loader.create_dataloaders(args.batch_size)
    t1_model, dwi_model = (model.to(device) for model in model_pair(cnn))
    init_paths = (args.init_t1w, args.init_dwi)
    for model, path in zip((t1_model, dwi_model), init_paths, strict=True):
        if path is not None:
            load_model_safe(model, path, device)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t1_optimizer = torch.optim.Adam(t1_model.parameters(), lr=args.learning_rate)
    dwi_optimizer = torch.optim.Adam(dwi_model.parameters(), lr=args.learning_rate)
    best_t1, best_dwi = float("inf"), float("inf")
    history = {metric: [] for metric in ("train_t1w", "val_t1w", "train_dwi", "val_dwi")}
    t1_path, dwi_path = paths
    for epoch in range(1, args.epochs + 1):
        t1_model.requires_grad_(True)
        t1_train = run_epoch(t1_model, train_loader, device, t1_optimizer)
        t1_val = run_epoch(t1_model, val_loader, device)
        if t1_val < best_t1:
            best_t1 = t1_val
            torch.save(t1_model.state_dict(), t1_path)
        if not t1_path.is_file():
            raise RuntimeError("T1w validation did not produce a finite checkpoint")
        t1_model.load_state_dict(torch.load(t1_path, map_location=device, weights_only=True))
        t1_model.eval()
        t1_model.requires_grad_(False)
        dwi_train = run_epoch(dwi_model, train_loader, device, dwi_optimizer, t1_model)
        dwi_val = run_epoch(dwi_model, val_loader, device, t1_model=t1_model)
        if dwi_val < best_dwi:
            best_dwi = dwi_val
            torch.save(dwi_model.state_dict(), dwi_path)
        if not dwi_path.is_file():
            raise RuntimeError("DWI validation did not produce a finite checkpoint")
        for key, value in zip(history, (t1_train, t1_val, dwi_train, dwi_val), strict=True):
            history[key].append(value)
        savemat(history_path, history)
        logging.info("Epoch %d/%d  T1w train/val %.5g / %.5g  DWI train/val %.5g / %.5g",
                     epoch, args.epochs, t1_train, t1_val, dwi_train, dwi_val)


if __name__ == "__main__":
    main()
