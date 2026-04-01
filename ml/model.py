"""
CongestionLSTM — predicts probability of congestion from a sliding
window of telemetry features.

Input  : (batch, seq_len, 15)   [5 features × 3 slices per timestep]
Output : (batch, 1)             congestion probability [0, 1]
"""
import torch
import torch.nn as nn
from typing import Tuple


INPUT_FEATURES = 15   # 5 metrics × 3 slices
HIDDEN_SIZE = 64
NUM_LAYERS = 2


class CongestionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


# ── Synthetic training data ──────────────────────────────────────────────────

def _normal_sample(seq_len: int) -> Tuple[list, float]:
    """Low latency, low loss → label 0.0 (no congestion)."""
    import random
    row = []
    for _ in range(seq_len):
        crit = [random.uniform(0.01, 0.05), random.uniform(0.5, 0.7),
                random.uniform(0.0, 0.02), random.uniform(0.0, 0.02),
                random.uniform(0.55, 0.65)]
        std  = [random.uniform(0.02, 0.08), random.uniform(0.25, 0.35),
                random.uniform(0.0, 0.03), random.uniform(0.0, 0.03),
                random.uniform(0.25, 0.35)]
        bg   = [random.uniform(0.03, 0.10), random.uniform(0.05, 0.12),
                random.uniform(0.0, 0.04), random.uniform(0.0, 0.04),
                random.uniform(0.08, 0.12)]
        row.append(crit + std + bg)
    return row, 0.05


def _congestion_sample(seq_len: int) -> Tuple[list, float]:
    """High latency, high loss, queues building → label 0.85–1.0."""
    import random
    row = []
    ramp = 0.0
    for t in range(seq_len):
        ramp = min(1.0, ramp + 0.05)
        crit = [random.uniform(0.4, 0.9) * ramp, random.uniform(0.7, 1.0),
                random.uniform(0.05, 0.3) * ramp, random.uniform(0.05, 0.2) * ramp,
                random.uniform(0.55, 0.65)]
        std  = [random.uniform(0.3, 0.7) * ramp, random.uniform(0.3, 0.5),
                random.uniform(0.03, 0.15) * ramp, random.uniform(0.03, 0.1) * ramp,
                random.uniform(0.25, 0.35)]
        bg   = [random.uniform(0.2, 0.6) * ramp, random.uniform(0.1, 0.25),
                random.uniform(0.02, 0.2) * ramp, random.uniform(0.02, 0.15) * ramp,
                random.uniform(0.08, 0.12)]
        row.append(crit + std + bg)
    label = random.uniform(0.80, 0.98)
    return row, label


def _transition_sample(seq_len: int) -> Tuple[list, float]:
    """Transition from normal to congested (label 0.4–0.7)."""
    import random
    half = seq_len // 2
    normal_rows = [r for r, _ in [_normal_sample(1) for _ in range(half)]]
    cong_rows = [r for r, _ in [_congestion_sample(1) for _ in range(seq_len - half)]]
    combined = [r[0] for r in normal_rows] + [r[0] for r in cong_rows]
    return combined, random.uniform(0.40, 0.70)


def generate_training_data(
    n_samples: int = 2000,
    seq_len: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import random
    xs, ys = [], []
    generators = [_normal_sample, _congestion_sample, _transition_sample]
    weights = [0.45, 0.35, 0.20]
    for _ in range(n_samples):
        gen = random.choices(generators, weights=weights, k=1)[0]
        x, y = gen(seq_len)
        xs.append(x)
        ys.append([y])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def train_model(
    seq_len: int = 50,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> CongestionLSTM:
    import random
    torch.manual_seed(42)
    random.seed(42)

    model = CongestionLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    X, y = generate_training_data(2400, seq_len)
    X, y = X.to(device), y.to(device)

    # 80/20 train/val split for early stopping
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    n = len(X_train)
    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n)
        X_train, y_train = X_train[idx], y_train[idx]
        for i in range(0, n, batch_size):
            xb = X_train[i:i + batch_size]
            yb = y_train[i:i + batch_size]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation loss for early stopping
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        model.train()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        scheduler.step()

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model
