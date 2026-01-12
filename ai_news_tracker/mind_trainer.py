"""
MIND Dataset Training Module.

Implements the NRMS (Neural News Recommendation with Multi-Head Self-Attention) model
for training on the Microsoft News Dataset (MIND).

Reference: https://aclanthology.org/2020.acl-main.331/
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from urllib.request import urlretrieve
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# MIND dataset URLs
MIND_URLS = {
    "small": {
        "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
    },
    "large": {
        "train": "https://mind201910.blob.core.windows.net/release/MINDlarge_train.zip",
        "dev": "https://mind201910.blob.core.windows.net/release/MINDlarge_dev.zip",
        "test": "https://mind201910.blob.core.windows.net/release/MINDlarge_test.zip",
    },
}


@dataclass
class MINDConfig:
    """Configuration for MIND training."""
    # Data settings
    data_dir: Path = field(default_factory=lambda: Path.home() / ".news_tracker" / "mind")
    dataset_size: str = "small"  # "small" or "large"
    max_title_len: int = 30
    max_history_len: int = 50

    # Model settings
    embedding_dim: int = 256  # Must be divisible by num_attention_heads
    num_attention_heads: int = 16
    attention_dim: int = 256  # Must be divisible by num_attention_heads
    dropout: float = 0.2

    # Training settings
    batch_size: int = 64
    learning_rate: float = 0.0001
    epochs: int = 5
    negative_samples: int = 4  # Negative samples per positive click


@dataclass
class NewsArticle:
    """Represents a news article from MIND."""
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    title_tokens: List[int] = field(default_factory=list)


@dataclass
class UserBehavior:
    """Represents a user's behavior (impression)."""
    impression_id: str
    user_id: str
    history: List[str]  # List of news IDs in history
    impressions: List[Tuple[str, int]]  # (news_id, clicked)


class MINDDataset:
    """Handles MIND dataset loading and preprocessing."""

    def __init__(self, config: MINDConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.news: Dict[str, NewsArticle] = {}
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    def download(self, split: str = "train") -> Path:
        """Download MIND dataset if not present."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        urls = MIND_URLS.get(self.config.dataset_size, MIND_URLS["small"])
        if split not in urls:
            raise ValueError(f"Split '{split}' not available for {self.config.dataset_size}")

        url = urls[split]
        zip_name = f"MIND{self.config.dataset_size}_{split}.zip"
        zip_path = self.data_dir / zip_name
        extract_dir = self.data_dir / f"{self.config.dataset_size}_{split}"

        if extract_dir.exists():
            print(f"Dataset already exists at {extract_dir}")
            return extract_dir

        if not zip_path.exists():
            print(f"Downloading {zip_name}...")
            urlretrieve(url, zip_path, reporthook=_download_progress)
            print()

        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        return extract_dir

    def load_news(self, data_path: Path) -> Dict[str, NewsArticle]:
        """Load news articles from news.tsv."""
        news_file = data_path / "news.tsv"
        news = {}

        with open(news_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    news_id, category, subcategory, title, abstract = parts[:5]
                    news[news_id] = NewsArticle(
                        news_id=news_id,
                        category=category,
                        subcategory=subcategory,
                        title=title,
                        abstract=abstract or "",
                    )

        return news

    def load_behaviors(self, data_path: Path) -> List[UserBehavior]:
        """Load user behaviors from behaviors.tsv."""
        behaviors_file = data_path / "behaviors.tsv"
        behaviors = []

        with open(behaviors_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    imp_id, user_id, _, history_str, impressions_str = parts[:5]

                    # Parse history (may be empty)
                    history = history_str.split() if history_str else []

                    # Parse impressions (news_id-clicked pairs)
                    impressions = []
                    for imp in impressions_str.split():
                        if "-" in imp:
                            news_id, clicked = imp.rsplit("-", 1)
                            impressions.append((news_id, int(clicked)))

                    if impressions:  # Only add if there are impressions
                        behaviors.append(UserBehavior(
                            impression_id=imp_id,
                            user_id=user_id,
                            history=history,
                            impressions=impressions,
                        ))

        return behaviors

    def build_vocab(self, news: Dict[str, NewsArticle], min_freq: int = 2):
        """Build vocabulary from news titles."""
        word_counts: Dict[str, int] = {}

        for article in news.values():
            for word in article.title.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Add words meeting frequency threshold
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")

    def tokenize_title(self, title: str) -> List[int]:
        """Convert title to token indices."""
        tokens = []
        for word in title.lower().split()[:self.config.max_title_len]:
            tokens.append(self.word2idx.get(word, self.word2idx["<UNK>"]))

        # Pad to max length
        while len(tokens) < self.config.max_title_len:
            tokens.append(self.word2idx["<PAD>"])

        return tokens

    def tokenize_all_news(self):
        """Tokenize all news titles."""
        for article in self.news.values():
            article.title_tokens = self.tokenize_title(article.title)

    def save_vocab(self, path: Path):
        """Save vocabulary to disk."""
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx}, f)

    def load_vocab(self, path: Path):
        """Load vocabulary from disk."""
        with open(path) as f:
            data = json.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}


def _download_progress(count, block_size, total_size):
    """Progress callback for urlretrieve."""
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading: {percent}%", end="", flush=True)


if TORCH_AVAILABLE:

    class MultiHeadSelfAttention(nn.Module):
        """Multi-head self-attention layer."""

        def __init__(self, input_dim: int, num_heads: int, attention_dim: int):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = attention_dim // num_heads

            self.query = nn.Linear(input_dim, attention_dim)
            self.key = nn.Linear(input_dim, attention_dim)
            self.value = nn.Linear(input_dim, attention_dim)
            self.output = nn.Linear(attention_dim, input_dim)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            batch_size, seq_len, _ = x.shape

            # Project to Q, K, V
            q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

            attn = F.softmax(scores, dim=-1)

            # Apply attention to values
            context = torch.matmul(attn, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

            return self.output(context)

    class AdditiveAttention(nn.Module):
        """Additive attention for aggregating word/news representations."""

        def __init__(self, input_dim: int, attention_dim: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(input_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1, bias=False),
            )

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # x: (batch, seq_len, dim)
            scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
            return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, dim)

    class NewsEncoder(nn.Module):
        """
        Encodes news articles using multi-head self-attention.

        Architecture:
        - Word embedding
        - Multi-head self-attention
        - Additive attention for aggregation
        """

        def __init__(self, config: MINDConfig, vocab_size: int):
            super().__init__()
            self.config = config

            self.word_embedding = nn.Embedding(
                vocab_size,
                config.embedding_dim,
                padding_idx=0,
            )

            self.self_attention = MultiHeadSelfAttention(
                config.embedding_dim,
                config.num_attention_heads,
                config.attention_dim,
            )

            self.additive_attention = AdditiveAttention(
                config.embedding_dim,
                config.attention_dim,
            )

            self.dropout = nn.Dropout(config.dropout)

        def forward(self, title_tokens: torch.Tensor) -> torch.Tensor:
            """
            Args:
                title_tokens: (batch, max_title_len) token indices

            Returns:
                News embeddings (batch, embedding_dim)
            """
            # Create mask for padding
            mask = (title_tokens != 0).float()

            # Embed words
            x = self.word_embedding(title_tokens)  # (batch, seq, dim)
            x = self.dropout(x)

            # Self-attention
            x = self.self_attention(x, mask)
            x = self.dropout(x)

            # Aggregate with additive attention
            news_repr = self.additive_attention(x, mask)

            return news_repr

    class UserEncoder(nn.Module):
        """
        Encodes user preferences from browsing history.

        Architecture:
        - News encoder for each article in history
        - Multi-head self-attention over history
        - Additive attention for aggregation
        """

        def __init__(self, config: MINDConfig, news_encoder: NewsEncoder):
            super().__init__()
            self.config = config
            self.news_encoder = news_encoder

            self.self_attention = MultiHeadSelfAttention(
                config.embedding_dim,
                config.num_attention_heads,
                config.attention_dim,
            )

            self.additive_attention = AdditiveAttention(
                config.embedding_dim,
                config.attention_dim,
            )

            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            history_tokens: torch.Tensor,
            history_mask: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                history_tokens: (batch, history_len, max_title_len)
                history_mask: (batch, history_len) - 1 for real, 0 for padding

            Returns:
                User embeddings (batch, embedding_dim)
            """
            batch_size, history_len, title_len = history_tokens.shape

            # Flatten for news encoder
            flat_tokens = history_tokens.view(-1, title_len)
            flat_repr = self.news_encoder(flat_tokens)

            # Reshape back
            history_repr = flat_repr.view(batch_size, history_len, -1)

            # Self-attention over history
            history_repr = self.self_attention(history_repr, history_mask)
            history_repr = self.dropout(history_repr)

            # Aggregate with additive attention
            user_repr = self.additive_attention(history_repr, history_mask)

            return user_repr

    class NRMSModel(nn.Module):
        """
        Neural News Recommendation with Multi-Head Self-Attention (NRMS).

        Reference: Wu et al., EMNLP 2019
        """

        def __init__(self, config: MINDConfig, vocab_size: int):
            super().__init__()
            self.config = config

            self.news_encoder = NewsEncoder(config, vocab_size)
            self.user_encoder = UserEncoder(config, self.news_encoder)

        def forward(
            self,
            candidate_tokens: torch.Tensor,
            history_tokens: torch.Tensor,
            history_mask: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                candidate_tokens: (batch, num_candidates, max_title_len)
                history_tokens: (batch, history_len, max_title_len)
                history_mask: (batch, history_len)

            Returns:
                Click probabilities (batch, num_candidates)
            """
            batch_size, num_candidates, title_len = candidate_tokens.shape

            # Encode user
            user_repr = self.user_encoder(history_tokens, history_mask)  # (batch, dim)

            # Encode candidates
            flat_candidates = candidate_tokens.view(-1, title_len)
            flat_repr = self.news_encoder(flat_candidates)
            candidate_repr = flat_repr.view(batch_size, num_candidates, -1)

            # Dot product scoring
            scores = torch.bmm(candidate_repr, user_repr.unsqueeze(-1)).squeeze(-1)

            return scores

        def get_news_embedding(self, title_tokens: torch.Tensor) -> torch.Tensor:
            """Get embedding for a single news article."""
            return self.news_encoder(title_tokens)

        def get_user_embedding(
            self,
            history_tokens: torch.Tensor,
            history_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Get embedding for a user based on history."""
            return self.user_encoder(history_tokens, history_mask)

    class MINDTrainingDataset(Dataset):
        """PyTorch dataset for MIND training."""

        def __init__(
            self,
            behaviors: List[UserBehavior],
            news: Dict[str, NewsArticle],
            config: MINDConfig,
            negative_samples: int = 4,
        ):
            self.config = config
            self.news = news
            self.negative_samples = negative_samples

            # Build training samples
            self.samples = []
            for behavior in behaviors:
                if not behavior.history:
                    continue

                # Get clicked and non-clicked news
                clicked = [nid for nid, c in behavior.impressions if c == 1]
                non_clicked = [nid for nid, c in behavior.impressions if c == 0]

                if not clicked or not non_clicked:
                    continue

                for pos_id in clicked:
                    if pos_id not in news:
                        continue

                    # Sample negatives
                    valid_negs = [nid for nid in non_clicked if nid in news]
                    if len(valid_negs) < negative_samples:
                        continue

                    neg_ids = random.sample(valid_negs, negative_samples)
                    valid_history = [nid for nid in behavior.history if nid in news]

                    if valid_history:
                        self.samples.append((valid_history, pos_id, neg_ids))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            history_ids, pos_id, neg_ids = self.samples[idx]

            # Get history tokens
            history = history_ids[-self.config.max_history_len:]
            history_tokens = np.zeros((self.config.max_history_len, self.config.max_title_len), dtype=np.int64)
            history_mask = np.zeros(self.config.max_history_len, dtype=np.float32)

            for i, nid in enumerate(history):
                if nid in self.news:
                    history_tokens[i] = self.news[nid].title_tokens
                    history_mask[i] = 1.0

            # Get candidate tokens (positive + negatives)
            num_candidates = 1 + self.negative_samples
            candidate_tokens = np.zeros((num_candidates, self.config.max_title_len), dtype=np.int64)

            candidate_tokens[0] = self.news[pos_id].title_tokens
            for i, neg_id in enumerate(neg_ids):
                candidate_tokens[i + 1] = self.news[neg_id].title_tokens

            # Labels: first is positive
            labels = np.zeros(num_candidates, dtype=np.float32)
            labels[0] = 1.0

            return {
                "history_tokens": torch.tensor(history_tokens),
                "history_mask": torch.tensor(history_mask),
                "candidate_tokens": torch.tensor(candidate_tokens),
                "labels": torch.tensor(labels),
            }


class MINDTrainer:
    """Orchestrates MIND dataset training."""

    def __init__(self, config: Optional[MINDConfig] = None):
        self.config = config or MINDConfig()
        self.dataset = MINDDataset(self.config)
        self.model: Optional[NRMSModel] = None

    def setup(self, download: bool = True) -> None:
        """Download and prepare the dataset."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for MIND training")

        if download:
            train_path = self.dataset.download("train")
            dev_path = self.dataset.download("dev")
        else:
            train_path = self.config.data_dir / f"{self.config.dataset_size}_train"
            dev_path = self.config.data_dir / f"{self.config.dataset_size}_dev"

        # Load news
        print("Loading news articles...")
        train_news = self.dataset.load_news(train_path)
        dev_news = self.dataset.load_news(dev_path)
        self.dataset.news = {**train_news, **dev_news}
        print(f"Loaded {len(self.dataset.news)} news articles")

        # Build vocabulary
        print("Building vocabulary...")
        self.dataset.build_vocab(self.dataset.news)

        # Tokenize all news
        print("Tokenizing news...")
        self.dataset.tokenize_all_news()

        # Save vocab for later use
        vocab_path = self.config.data_dir / "vocab.json"
        self.dataset.save_vocab(vocab_path)

    def train(self, verbose: bool = True) -> dict:
        """Train the NRMS model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training")

        train_path = self.config.data_dir / f"{self.config.dataset_size}_train"
        dev_path = self.config.data_dir / f"{self.config.dataset_size}_dev"

        # Load behaviors
        print("Loading training behaviors...")
        train_behaviors = self.dataset.load_behaviors(train_path)
        print(f"Loaded {len(train_behaviors)} training impressions")

        print("Loading validation behaviors...")
        dev_behaviors = self.dataset.load_behaviors(dev_path)
        print(f"Loaded {len(dev_behaviors)} validation impressions")

        # Create datasets
        train_dataset = MINDTrainingDataset(
            train_behaviors,
            self.dataset.news,
            self.config,
            self.config.negative_samples,
        )
        print(f"Training samples: {len(train_dataset)}")

        dev_dataset = MINDTrainingDataset(
            dev_behaviors[:10000],  # Limit dev size
            self.dataset.news,
            self.config,
            self.config.negative_samples,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Initialize model
        vocab_size = len(self.dataset.word2idx)
        self.model = NRMSModel(self.config, vocab_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f"Training on {device}")

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_auc = 0.0
        best_state = None

        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                history_tokens = batch["history_tokens"].to(device)
                history_mask = batch["history_mask"].to(device)
                candidate_tokens = batch["candidate_tokens"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                scores = self.model(candidate_tokens, history_tokens, history_mask)

                # Cross entropy loss (first position is positive)
                loss = criterion(scores, torch.zeros(scores.size(0), dtype=torch.long, device=device))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches

            # Validation
            auc = self._evaluate(dev_loader, device)

            if verbose:
                print(f"Epoch {epoch + 1}/{self.config.epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        return {
            "best_auc": best_auc,
            "epochs": self.config.epochs,
            "vocab_size": vocab_size,
        }

    def _evaluate(self, loader: DataLoader, device: torch.device) -> float:
        """Evaluate model and return AUC."""
        assert self.model is not None, "Model must be initialized before evaluation"
        self.model.eval()
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for batch in loader:
                history_tokens = batch["history_tokens"].to(device)
                history_mask = batch["history_mask"].to(device)
                candidate_tokens = batch["candidate_tokens"].to(device)
                labels = batch["labels"]

                scores = self.model(candidate_tokens, history_tokens, history_mask)
                scores = torch.softmax(scores, dim=-1)

                all_labels.extend(labels.numpy().flatten().tolist())
                all_scores.extend(scores.cpu().numpy().flatten().tolist())

        # Calculate AUC
        return self._calculate_auc(all_labels, all_scores)

    def _calculate_auc(self, labels: List[float], scores: List[float]) -> float:
        """Calculate AUC-ROC score."""
        pairs = list(zip(scores, labels))
        pairs.sort(reverse=True)

        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp = 0
        fp = 0
        auc = 0.0
        prev_score = None

        for score, label in pairs:
            if prev_score is not None and score != prev_score:
                pass  # Score changed
            if label == 1:
                tp += 1
            else:
                auc += tp
                fp += 1
            prev_score = score

        return auc / (n_pos * n_neg) if n_pos * n_neg > 0 else 0.5

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save - train first!")

        path = path or (self.config.data_dir / "nrms_model.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "num_attention_heads": self.config.num_attention_heads,
                "attention_dim": self.config.attention_dim,
                "max_title_len": self.config.max_title_len,
                "max_history_len": self.config.max_history_len,
            },
            "vocab_size": len(self.dataset.word2idx),
        }, path)

        return path

    def load_model(self, path: Optional[Path] = None) -> None:
        """Load trained model from disk."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")

        path = path or (self.config.data_dir / "nrms_model.pt")

        checkpoint = torch.load(path, weights_only=True)
        vocab_size = checkpoint["vocab_size"]

        self.model = NRMSModel(self.config, vocab_size)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # Load vocab
        vocab_path = self.config.data_dir / "vocab.json"
        if vocab_path.exists():
            self.dataset.load_vocab(vocab_path)
