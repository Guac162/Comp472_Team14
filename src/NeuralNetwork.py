import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data import load
from src.preprocess import preprocess
from src.utils import set_seed, accuracy, timer, log_result

WHO = "Yousra"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- model ----------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation=nn.ReLU, dropout=0.0):
        super().__init__()
        layers, in_features = [], input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)


# ---------- train / eval / predict ----------
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def train(model, train_loader, val_loader, lr=1e-3, weight_decay=0.0,
          max_epochs=100, patience=5):
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc, best_state, wait = -1.0, None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(max_epochs):
        model.train()
        running_loss, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            n += len(y)

        train_loss = running_loss / n
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, DEVICE)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1:3d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, wait = val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch + 1
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history, best_acc


def predict(model, X):
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X.astype("float32")).to(DEVICE)
        return model(t).argmax(dim=1).cpu().numpy()


def make_loaders(X_tr, y_tr, X_va, y_va, batch_size=128):
    tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    return DataLoader(tr, batch_size=batch_size, shuffle=True), DataLoader(va, batch_size=batch_size)


# ---------- one dataset: run configs, log each, keep best ----------
def run_dataset(name, configs):
    X_tr0, y_tr, X_va0, y_va, X_te0, ids = load(name)
    n_classes = int(y_tr.max()) + 1

    best_acc, best_cfg = -1.0, None
    for cfg in configs:
        print(f"\n=== {name}: {cfg} ===")
        set_seed()   # same seed for every config, so runs are comparable

        # scaler (if any) is fitted on train only, then applied to val and test
        X_tr, X_va, X_te = preprocess(X_tr0, X_va0, X_te0, method=cfg["method"])
        tr_loader, va_loader = make_loaders(X_tr, y_tr, X_va, y_va)

        model = NeuralNetwork(X_tr.shape[1], cfg["hidden_sizes"], n_classes,
                              dropout=cfg["dropout"])

        with timer() as t_train:
            model, history, _ = train(model, tr_loader, va_loader,
                                      lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        with timer() as t_pred:
            val_pred = predict(model, X_va)
            test_pred = predict(model, X_te)
        train_pred = predict(model, X_tr)   # not timed: only for train_acc

        train_acc = accuracy(y_tr, train_pred)
        val_acc = accuracy(y_va, val_pred)

        params = dict(cfg, n_classes=n_classes,
                      epochs_run=len(history["val_acc"]),
                      best_epoch=history["best_epoch"])

        # predictions are saved only when this config is the best so far, so
        # preds/*/nn_<dataset>.npy always ends up holding the best config.
        is_best = val_acc > best_acc
        log_result(model="nn", dataset=name, who=WHO, params=params,
                   train_acc=train_acc, val_acc=val_acc,
                   train_time_s=t_train["s"], predict_time_s=t_pred["s"],
                   val_pred=val_pred if is_best else None,
                   test_pred=test_pred if is_best else None)

        if is_best:
            best_acc, best_cfg = val_acc, cfg

    print(f"\nBest {name} val acc {best_acc:.4f} with {best_cfg}")
    return best_acc


if __name__ == "__main__":
    sent_configs = [
        dict(method="raw", hidden_sizes=[512, 128], dropout=0.1, lr=1e-3, weight_decay=1e-3),
    ]
    img_configs = [
        dict(method="raw", hidden_sizes=[512, 128], dropout=0.1, lr=1e-3, weight_decay=1e-5),
    ]

    run_dataset("sentiment", sent_configs)
    run_dataset("image", img_configs)