"""Lightweight neural re-ranker trained on user feedback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class RerankerConfig:
    """Configuration for the re-ranker model."""
    embedding_dim: int = 384
    hidden_dim: int = 128
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    min_samples: int = 10  # Minimum feedback samples before training


class FeedbackDataset:
    """Dataset of user feedback for training the re-ranker."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.samples: List[Tuple[np.ndarray, np.ndarray, float]] = []

    def add_sample(
        self,
        article_embedding: np.ndarray,
        user_embedding: np.ndarray,
        label: float,  # 1.0 for like, 0.0 for dislike
    ):
        """Add a feedback sample."""
        self.samples.append((
            article_embedding.astype(np.float32),
            user_embedding.astype(np.float32),
            label,
        ))

    def __len__(self):
        return len(self.samples)

    def to_tensors(self):
        """Convert to PyTorch tensors."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for re-ranker training")

        if not self.samples:
            return None, None, None

        article_embs = torch.tensor([s[0] for s in self.samples])
        user_embs = torch.tensor([s[1] for s in self.samples])
        labels = torch.tensor([s[2] for s in self.samples], dtype=torch.float32)

        return article_embs, user_embs, labels

    def save(self, path: Path):
        """Save dataset to disk."""
        data = {
            "embedding_dim": self.embedding_dim,
            "samples": [
                {
                    "article": s[0].tolist(),
                    "user": s[1].tolist(),
                    "label": s[2],
                }
                for s in self.samples
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "FeedbackDataset":
        """Load dataset from disk."""
        with open(path) as f:
            data = json.load(f)

        dataset = cls(embedding_dim=data["embedding_dim"])
        for s in data["samples"]:
            dataset.samples.append((
                np.array(s["article"], dtype=np.float32),
                np.array(s["user"], dtype=np.float32),
                s["label"],
            ))
        return dataset


if TORCH_AVAILABLE:
    class RerankerModel(nn.Module):
        """
        Simple MLP re-ranker that learns to predict user preference.

        Takes concatenated [article_embedding, user_embedding] as input
        and outputs a relevance score.
        """

        def __init__(self, config: RerankerConfig):
            super().__init__()
            self.config = config

            input_dim = config.embedding_dim * 2  # article + user embeddings

            self.network = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, article_emb: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                article_emb: (batch, embedding_dim)
                user_emb: (batch, embedding_dim)

            Returns:
                Relevance scores (batch,)
            """
            combined = torch.cat([article_emb, user_emb], dim=-1)
            return self.network(combined).squeeze(-1)

        def predict(self, article_emb: np.ndarray, user_emb: np.ndarray) -> float:
            """Predict relevance score for a single article."""
            self.eval()
            with torch.no_grad():
                article_t = torch.tensor(article_emb, dtype=torch.float32).unsqueeze(0)
                user_t = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0)
                score = self.forward(article_t, user_t)
                return float(score.item())

        def predict_batch(
            self,
            article_embs: np.ndarray,
            user_emb: np.ndarray,
        ) -> np.ndarray:
            """Predict relevance scores for multiple articles."""
            self.eval()
            with torch.no_grad():
                article_t = torch.tensor(article_embs, dtype=torch.float32)
                # Broadcast user embedding to batch size
                user_t = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0)
                user_t = user_t.expand(article_t.shape[0], -1)
                scores = self.forward(article_t, user_t)
                return scores.numpy()
else:
    # Dummy class when PyTorch not available
    class RerankerModel:
        def __init__(self, config: RerankerConfig):
            raise RuntimeError("PyTorch is required for the re-ranker model")


class Reranker:
    """
    Manages the re-ranker model lifecycle: training, saving, loading, inference.
    """

    def __init__(
        self,
        model_dir: Path,
        config: Optional[RerankerConfig] = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or RerankerConfig()

        self.model: Optional[RerankerModel] = None
        self.dataset = FeedbackDataset(self.config.embedding_dim)
        self.is_trained = False

        # Try to load existing model and dataset
        self._load_state()

    def _load_state(self):
        """Load model and dataset from disk if available."""
        dataset_path = self.model_dir / "feedback_dataset.json"
        model_path = self.model_dir / "reranker_model.pt"
        config_path = self.model_dir / "reranker_config.json"

        # Load config
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                self.config = RerankerConfig(**config_data)

        # Load dataset
        if dataset_path.exists():
            self.dataset = FeedbackDataset.load(dataset_path)

        # Load model
        if model_path.exists() and TORCH_AVAILABLE:
            self.model = RerankerModel(self.config)
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.model.eval()
            self.is_trained = True

    def save_state(self):
        """Save model and dataset to disk."""
        # Save config
        config_path = self.model_dir / "reranker_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "embedding_dim": self.config.embedding_dim,
                "hidden_dim": self.config.hidden_dim,
                "dropout": self.config.dropout,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "min_samples": self.config.min_samples,
            }, f)

        # Save dataset
        dataset_path = self.model_dir / "feedback_dataset.json"
        self.dataset.save(dataset_path)

        # Save model
        if self.model is not None and TORCH_AVAILABLE:
            model_path = self.model_dir / "reranker_model.pt"
            torch.save(self.model.state_dict(), model_path)

    def add_feedback(
        self,
        article_embedding: np.ndarray,
        user_embedding: np.ndarray,
        liked: bool,
    ):
        """
        Add a user feedback sample.

        Args:
            article_embedding: The article's embedding vector
            user_embedding: The user's preference embedding
            liked: True if user liked, False if disliked
        """
        label = 1.0 if liked else 0.0
        self.dataset.add_sample(article_embedding, user_embedding, label)
        self.save_state()

    def can_train(self) -> bool:
        """Check if we have enough samples to train."""
        return len(self.dataset) >= self.config.min_samples

    def train(self, verbose: bool = True) -> dict:
        """
        Train the re-ranker model on collected feedback.

        Returns:
            Training metrics (loss, accuracy)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training")

        if not self.can_train():
            raise ValueError(
                f"Need at least {self.config.min_samples} samples to train, "
                f"but only have {len(self.dataset)}"
            )

        # Prepare data
        article_embs, user_embs, labels = self.dataset.to_tensors()

        # Split into train/val (80/20)
        n = len(labels)
        indices = torch.randperm(n)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]

        # Initialize model
        self.model = RerankerModel(self.config)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        best_state = None
        patience = 10
        no_improve = 0

        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            # Simple batch iteration
            for i in range(0, len(train_idx), self.config.batch_size):
                batch_idx = train_idx[i:i + self.config.batch_size]

                optimizer.zero_grad()
                preds = self.model(article_embs[batch_idx], user_embs[batch_idx])
                loss = criterion(preds, labels[batch_idx])
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(article_embs[val_idx], user_embs[val_idx])
                val_loss = criterion(val_preds, labels[val_idx]).item()

                # Accuracy
                val_acc = ((val_preds > 0.5) == labels[val_idx]).float().mean().item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.2%}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_trained = True
        self.save_state()

        # Final metrics
        with torch.no_grad():
            all_preds = self.model(article_embs, user_embs)
            final_acc = ((all_preds > 0.5) == labels).float().mean().item()

        return {
            "samples": len(self.dataset),
            "best_val_loss": best_val_loss,
            "final_accuracy": final_acc,
        }

    def rerank(
        self,
        articles_with_scores: List[Tuple],  # (article, score, freshness, embedding)
        user_embedding: np.ndarray,
        blend_weight: float = 0.5,  # How much to blend reranker vs original score
    ) -> List[Tuple]:
        """
        Re-rank articles using the trained model.

        Args:
            articles_with_scores: List of (article, original_score, freshness, embedding)
            user_embedding: User's preference embedding
            blend_weight: 0=only original, 1=only reranker

        Returns:
            Re-ranked list of (article, blended_score, freshness)
        """
        if not self.is_trained or self.model is None:
            # Fall back to original scores
            return [(a, s, f) for a, s, f, _ in articles_with_scores]

        # Get reranker predictions
        embeddings = np.array([e for _, _, _, e in articles_with_scores])
        reranker_scores = self.model.predict_batch(embeddings, user_embedding)

        # Blend scores
        results = []
        for i, (article, orig_score, freshness, _) in enumerate(articles_with_scores):
            blended = (1 - blend_weight) * orig_score + blend_weight * reranker_scores[i]
            results.append((article, float(blended), freshness))

        # Sort by blended score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_stats(self) -> dict:
        """Get reranker statistics."""
        return {
            "is_trained": self.is_trained,
            "feedback_samples": len(self.dataset),
            "can_train": self.can_train(),
            "min_samples_required": self.config.min_samples,
            "pytorch_available": TORCH_AVAILABLE,
        }
