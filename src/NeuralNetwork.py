import numpy as np
import pandas as pd
import copy
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader


sentimentsplit = np.load("sentiment_split.npz")
senttrain_idx, sentval_idx = sentimentsplit["train_idx"], sentimentsplit["val_idx"]

imagesplit = np.load("image_split.npz")
imgtrain_idx, imgval_idx = imagesplit["train_idx"], imagesplit["val_idx"]


#check split
print(senttrain_idx.dtype, senttrain_idx[:5])
print(len(senttrain_idx), len(sentval_idx))    

print(imgtrain_idx.dtype, imgtrain_idx[:5])
print(len(imgtrain_idx), len(imgval_idx))  

#load data
sentdata = pd.read_csv("sentiment_train.csv")
imgdata = pd.read_csv("image_train.csv")      

#split data according to npz file
senttrain_data = sentdata.iloc[senttrain_idx]
sentval_data   = sentdata.iloc[sentval_idx]

imgtrain_data = imgdata.iloc[imgtrain_idx]
imgval_data   = imgdata.iloc[imgval_idx]

#
assert set(senttrain_data["id"]).isdisjoint(sentval_data["id"])
assert len(senttrain_data) + len(sentval_data) == len(sentdata)

assert set(imgtrain_data["id"]).isdisjoint(imgval_data["id"])
assert len(imgtrain_data) + len(imgval_data) == len(imgdata)

#
X_senttrain = senttrain_data.drop(columns=["id", "label"]).values.astype("float32")
y_senttrain = senttrain_data["label"].values.astype("int64")
X_sentval   = sentval_data.drop(columns=["id", "label"]).values.astype("float32")
y_sentval   = sentval_data["label"].values.astype("int64")

X_imgtrain = imgtrain_data.drop(columns=["id", "label"]).values.astype("float32")
y_imgtrain = imgtrain_data["label"].values.astype("int64")
X_imgval   = imgval_data.drop(columns=["id", "label"]).values.astype("float32")
y_imgval   = imgval_data["label"].values.astype("int64")
#########################

#loaders
senttrain_ds = TensorDataset(torch.from_numpy(X_senttrain), torch.from_numpy(y_senttrain))
sentval_ds   = TensorDataset(torch.from_numpy(X_sentval),   torch.from_numpy(y_sentval))

senttrain_loader = DataLoader(senttrain_ds, batch_size=128, shuffle=True)
sentval_loader   = DataLoader(sentval_ds, batch_size=128)


imgtrain_ds = TensorDataset(torch.from_numpy(X_imgtrain), torch.from_numpy(y_imgtrain))
imgval_ds   = TensorDataset(torch.from_numpy(X_imgval),   torch.from_numpy(y_imgval))

imgtrain_loader = DataLoader(imgtrain_ds, batch_size=128, shuffle=True)
imgval_loader   = DataLoader(imgval_ds, batch_size=128)
################################

#NN class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation=nn.ReLU, dropout=0.0):
        super().__init__()

        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, output_size))

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
################################


#train and evaluate functions
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


def train(model, train_loader, val_loader, lr=0.001, max_epochs=100, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc, best_state, wait = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(max_epochs):
        model.train()
        running_loss, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            n += len(y)

        train_loss = running_loss / n
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:3d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, wait = val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)   # go back to the best weights
    return model, history, best_acc
###############################################################

#create model
sentmodel = NeuralNetwork(input_size=X_senttrain.shape[1], hidden_sizes=[256, 64], output_size=3)
print(sentmodel)
sentmodel, _, sent_acc = train(sentmodel, senttrain_loader, sentval_loader, lr=0.001, patience=5)

print("Best sentiment validation accuracy:", sent_acc)

imgmodel = NeuralNetwork(input_size=X_imgtrain.shape[1], hidden_sizes=[256, 64], output_size=50)
print(imgmodel)
imgmodel, _, img_acc = train(imgmodel, imgtrain_loader, imgval_loader)

print("Best image validation accuracy:", img_acc)


#prediction function
import pandas as pd
import torch

def predict_df(model, df, scaler=None):
    X = df.drop(columns=["id"]).values.astype("float32")
    if scaler is not None:
        X = scaler.transform(X).astype("float32")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        p = model(torch.from_numpy(X).to(device)).argmax(dim=1).cpu().numpy()
    return pd.DataFrame({"id": df["id"].values, "label": p})

sent_test = pd.read_csv("sentiment_test.csv")
img_test  = pd.read_csv("image_test.csv")

sent_out = predict_df(sentmodel, sent_test)
img_out  = predict_df(imgmodel, img_test)
##############################################

#check
sample = pd.read_csv("sample_submission.csv")
print(sample["id"].head(3).tolist())
print(sent_test["id"].head(3).tolist(), img_test["id"].head(3).tolist())

#prediction submission
preds_all = pd.concat([sent_out, img_out], ignore_index=True)

assert set(preds_all["id"]) == set(sample["id"]), "IDs don't match the template"

sub = sample[["id"]].merge(preds_all, on="id", how="left")   # keeps the template's row order
assert sub["label"].notna().all() and len(sub) == 10_000

sub.to_csv("predictions4.csv", index=False)
print(sub.head())
