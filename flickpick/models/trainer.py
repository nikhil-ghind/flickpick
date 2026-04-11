"""
Training loop for the two-tower retrieval model.
Supports full training and incremental updates with MLflow tracking.
"""

import logging
from dataclasses import dataclass

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flickpick.models.two_tower import TwoTowerModel

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    eval_every: int = 1000
    checkpoint_dir: str = "models/artifacts/two_tower"
    mlflow_experiment: str = "flickpick-two-tower"


class TwoTowerTrainer:
    def __init__(self, model: TwoTowerModel, config: TrainConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs,
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Full training loop with MLflow logging."""
        mlflow.set_experiment(self.config.mlflow_experiment)

        best_val_loss = float("inf")
        metrics = {}

        with mlflow.start_run():
            mlflow.log_params({
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "embedding_dim": self.model.user_tower.layer_norm.normalized_shape[0],
                "temperature": self.model.temperature,
            })

            global_step = 0
            for epoch in range(self.config.epochs):
                self.model.train()
                epoch_loss = 0.0
                num_batches = 0

                for batch in train_loader:
                    user_features, item_features = self._to_device(batch)

                    user_emb, item_emb = self.model(user_features, item_features)
                    loss = self.model.compute_loss(user_emb, item_emb)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                    global_step += 1

                    if global_step % self.config.eval_every == 0:
                        val_loss, val_metrics = self._evaluate(val_loader)
                        mlflow.log_metrics({
                            "val_loss": val_loss,
                            "val_recall@10": val_metrics["recall@10"],
                            "val_recall@50": val_metrics["recall@50"],
                        }, step=global_step)

                self.scheduler.step()

                avg_loss = epoch_loss / max(num_batches, 1)
                val_loss, val_metrics = self._evaluate(val_loader)

                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} — "
                    f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} "
                    f"recall@10={val_metrics['recall@10']:.4f}"
                )

                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_recall@10": val_metrics["recall@10"],
                    "val_recall@50": val_metrics["recall@50"],
                    "learning_rate": self.scheduler.get_last_lr()[0],
                }, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
                    metrics = {"best_epoch": epoch, "best_val_loss": val_loss, **val_metrics}

            mlflow.log_metrics({"best_val_loss": best_val_loss})

        return metrics

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> tuple[float, dict]:
        """Evaluate retrieval quality with recall@k."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_user_embs = []
        all_item_embs = []

        for batch in val_loader:
            user_features, item_features = self._to_device(batch)
            user_emb, item_emb = self.model(user_features, item_features)

            loss = self.model.compute_loss(user_emb, item_emb)
            total_loss += loss.item()
            num_batches += 1

            all_user_embs.append(user_emb)
            all_item_embs.append(item_emb)

        avg_loss = total_loss / max(num_batches, 1)

        # Compute recall@k on the full validation set
        user_embs = torch.cat(all_user_embs, dim=0)
        item_embs = torch.cat(all_item_embs, dim=0)
        sims = torch.matmul(user_embs, item_embs.T)
        labels = torch.arange(sims.size(0), device=sims.device)

        recall_10 = self._recall_at_k(sims, labels, k=10)
        recall_50 = self._recall_at_k(sims, labels, k=50)

        self.model.train()
        return avg_loss, {"recall@10": recall_10, "recall@50": recall_50}

    def _recall_at_k(self, sims: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        """Compute recall@k: fraction of queries where true item is in top-k."""
        _, topk_indices = sims.topk(k, dim=1)
        hits = (topk_indices == labels.unsqueeze(1)).any(dim=1).float()
        return hits.mean().item()

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = f"{self.config.checkpoint_dir}/model_epoch{epoch}_loss{val_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _to_device(self, batch):
        """Move batch tensors to device."""
        user_features = {k: v.to(self.device) for k, v in batch[0].items()}
        item_features = {k: v.to(self.device) for k, v in batch[1].items()}
        return user_features, item_features
