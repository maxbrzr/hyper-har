from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainerConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_delta: float = 0.0
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    checkpoint_path: str | None = "checkpoints/best_tinierhar.pt"
    early_stopping_metric: str = "val_loss"


@dataclass
class TrainerState:
    history: Dict[str, list[float]] = field(
        default_factory=lambda: {
            "train_loss": [],
            "train_macro_f1": [],
            "val_loss": [],
            "val_macro_f1": [],
        }
    )
    best_val_loss: float = float("inf")
    best_val_macro_f1: float = float("-inf")
    best_epoch: int = -1


class TinierHARTrainer:
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        config: TrainerConfig,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.num_classes = num_classes
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        weight = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.state = TrainerState()

    def _extract_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, Mapping):
            x = batch.get("x") or batch.get("features") or batch.get("inputs")
            y = (
                batch.get("y")
                or batch.get("label")
                or batch.get("labels")
                or batch.get("target")
            )
            if x is None or y is None:
                raise ValueError("Could not find input/target tensors in dict batch.")
            return x, y

        if isinstance(batch, Sequence) and len(batch) >= 2:
            a, b = batch[0], batch[1]
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                # Support both (x, y) and (y, x), e.g. TorchAdapter returns (y, x).
                if a.dim() <= 1 and b.dim() >= 2:
                    return b, a
                if b.dim() <= 1 and a.dim() >= 2:
                    return a, b
            return a, b

        raise ValueError(
            "Unsupported batch format. Expected dict or tuple/list with at least 2 items."
        )

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        # Expected by TinierHAR: (batch, 1, window_size, num_channels)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(
                f"Expected input with 3 or 4 dims, got shape {tuple(x.shape)}"
            )
        return x

    def _run_epoch(
        self, dataloader: DataLoader, train: bool, desc: str
    ) -> tuple[float, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        iterator = tqdm(dataloader, desc=desc, leave=False)
        for batch in iterator:
            x, y = self._extract_batch(batch)
            x = self._prepare_inputs(x).to(self.device).float()
            y = y.to(self.device).long().view(-1)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss = self.criterion(logits, y)
                if train:
                    loss.backward()
                    self.optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())
            running_correct += (preds == y).sum().item()
            running_total += batch_size

            iterator.set_postfix(
                loss=f"{(running_loss / max(1, running_total)):.4f}",
                acc=f"{(running_correct / max(1, running_total)):.4f}",
            )

        epoch_loss = running_loss / max(1, running_total)
        preds_t = torch.cat(all_preds).numpy()
        targets_t = torch.cat(all_targets).numpy()
        epoch_macro_f1 = f1_score(targets_t, preds_t, average="macro", zero_division=0)
        return epoch_loss, float(epoch_macro_f1)

    def _save_checkpoint(self) -> None:
        if not self.config.checkpoint_path:
            return
        ckpt_path = Path(self.config.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_path)

    def _load_checkpoint(self) -> None:
        if not self.config.checkpoint_path:
            return
        ckpt_path = Path(self.config.checkpoint_path)
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict[str, list[float]]:
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_macro_f1 = self._run_epoch(
                train_loader, train=True, desc=f"Train {epoch}/{self.config.epochs}"
            )
            val_loss, val_macro_f1 = self._run_epoch(
                val_loader, train=False, desc=f"Val {epoch}/{self.config.epochs}"
            )

            self.state.history["train_loss"].append(train_loss)
            self.state.history["train_macro_f1"].append(train_macro_f1)
            self.state.history["val_loss"].append(val_loss)
            self.state.history["val_macro_f1"].append(val_macro_f1)

            if self.config.early_stopping_metric == "val_macro_f1":
                improved = val_macro_f1 > (self.state.best_val_macro_f1 + self.config.min_delta)
            else:
                improved = val_loss < (self.state.best_val_loss - self.config.min_delta)
            if improved:
                self.state.best_val_loss = val_loss
                self.state.best_val_macro_f1 = val_macro_f1
                self.state.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} train_macro_f1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f} "
                f"best_val_loss={self.state.best_val_loss:.4f} "
                f"best_val_macro_f1={self.state.best_val_macro_f1:.4f} "
                f"patience={patience_counter}/{self.config.patience}"
            )

            if patience_counter >= self.config.patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best epoch: {self.state.best_epoch}."
                )
                break

        self._load_checkpoint()
        return self.state.history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Eval") -> Dict[str, Any]:
        self.model.eval()
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        running_loss = 0.0
        running_total = 0

        for batch in tqdm(dataloader, desc=desc, leave=False):
            x, y = self._extract_batch(batch)
            x = self._prepare_inputs(x).to(self.device).float()
            y = y.to(self.device).long().view(-1)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            running_total += y.size(0)

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

        preds_t = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)
        macro_f1 = f1_score(
            targets_t.numpy(), preds_t.numpy(), average="macro", zero_division=0
        )
        mean_loss = running_loss / max(1, running_total)
        cm = self.compute_confusion_matrix(targets_t, preds_t, self.num_classes)

        return {
            "loss": mean_loss,
            "macro_f1": float(macro_f1),
            "predictions": preds_t,
            "targets": targets_t,
            "confusion_matrix": cm,
        }

    @staticmethod
    def compute_confusion_matrix(
        targets: torch.Tensor, predictions: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for t, p in zip(targets.view(-1), predictions.view(-1)):
            cm[t.long(), p.long()] += 1
        return cm

    @staticmethod
    def save_confusion_matrix_plot(
        confusion_matrix: torch.Tensor,
        output_path: str,
        class_names: Iterable[str] | None = None,
        normalize: bool = True,
    ) -> None:
        cm = confusion_matrix.float()
        if normalize:
            row_sums = cm.sum(dim=1, keepdim=True).clamp_min(1.0)
            cm = cm / row_sums

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm.numpy(), interpolation="nearest")
        fig.colorbar(im, ax=ax)

        n_classes = cm.shape[0]
        tick_labels = (
            list(class_names)
            if class_names is not None
            else [str(i) for i in range(n_classes)]
        )
        ax.set(
            xlabel="Predicted label",
            ylabel="True label",
            xticks=list(range(n_classes)),
            yticks=list(range(n_classes)),
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
