import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import pytorch_lightning as pl

class CelebATripletDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        shift = random.randint(1, len(self.data.shape[0]) - 1)
        neg_idx = (idx + shift) % len(self.data.shape[0])
        return self.data[idx], self.data[idx], self.data[neg_idx]

class TripletModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.model(x)

class TripletModule(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, train_batch, batch_idx):
        result = self.model(train_batch)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        a, p, n = train_batch
        a_emb = self.model(a)
        n_emb = self.model(n)
        p_emb = self.model(p)
        cos = nn.CosineSimilarity(dim=1)
        loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x,y: -cos(x,y), margin=1/16)
        out_loss = loss(a_emb, p_emb, n_emb)
        self.log(
            "train_loss", out_loss.item()
        )
        return out_loss

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
          a, p, n = val_batch
          a_emb = self.model(a['image'])
          n_emb = self.model(n['image'])
          p_emb = self.model(p['image'])
          cos = nn.CosineSimilarity(dim=1)
          loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x,y: -cos(x,y), margin=1/16)
          out_loss = loss(a_emb, p_emb, n_emb)
          self.log(
              "val_loss", out_loss.item(), prog_bar=True
          )
          return out_loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 1e-5
    epochs = 20
    batch_size = 32

    model = TripletModel().to(device)
    module = TripletModule(model, learning_rate=learning_rate)

    trainer = pl.Trainer(accelerator="cpu", max_epochs=epochs)
    triplet_train_data = CelebATripletDataset('train')
    triplet_val_data = CelebATripletDataset('val')
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_data, batch_size=batch_size, shuffle=True)
    triplet_val_loader = torch.utils.data.DataLoader(triplet_val_data, batch_size=batch_size, shuffle=False)
    trainer.fit(module, triplet_train_loader, triplet_val_loader)
