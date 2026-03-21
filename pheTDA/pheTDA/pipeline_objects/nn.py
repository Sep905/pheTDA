import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
#from tqdm import tqdm


class AutoEncoder(nn.Module):

    def __init__(self,
                 input_dim, num_layers, use_batchnorm,   use_dropout,   dropout_prob, activation_function, learning_rate, w_decay, batch_size, epochs, random_state):

        super(AutoEncoder, self).__init__()

        torch.manual_seed(random_state)

        encoder_layers: list[nn.Module] = []
        decoder_layers: list[nn.Module] = []

        # ================= ENCODER =================
        start_dim = input_dim
        for i in range(num_layers-1):
            next_dim = int(start_dim / 2)
            encoder_layers.append(nn.Linear(start_dim, next_dim))
            
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(next_dim))

            if activation_function == "ReLU":
                encoder_layers.append(nn.ReLU())
            elif activation_function == "sigmoid":
                encoder_layers.append(nn.Sigmoid())
            elif activation_function == "tanh":
                encoder_layers.append(nn.Tanh())

            if use_dropout:
                encoder_layers.append(nn.Dropout(dropout_prob))
                
            start_dim = next_dim

        # Final encoder bottleneck layer (to 2D) - No activation/dropout here
        encoder_layers.append(nn.Linear(start_dim, 2))


        # ================= DECODER =================
        dec_dim = 2
        for i in range(num_layers-1):
            next_dim = dec_dim * 2
            decoder_layers.append(nn.Linear(dec_dim, next_dim))
            
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(next_dim))

            if activation_function == "ReLU":
                decoder_layers.append(nn.ReLU())
            elif activation_function == "sigmoid":
                decoder_layers.append(nn.Sigmoid())
            elif activation_function == "tanh":
                decoder_layers.append(nn.Tanh()) # Typo fixed here

            if use_dropout:
                decoder_layers.append(nn.Dropout(dropout_prob))
                
            dec_dim = next_dim

        # Final decoder output layer (back to input_dim) - No activation/dropout here
        decoder_layers.append(nn.Linear(dec_dim, input_dim))


        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=w_decay)

    def forward(self, x):

        hidden_representation = self.encoder(x)
        reconstruction = self.decoder(hidden_representation)
        return reconstruction,hidden_representation


    def fit_transform(self, x):
        self.train()

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy(dtype=np.float32)  # or x.values.astype(np.float32)
        elif isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        else:
            # assume tensor
            x = x.float()

        x = torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x
        x = x.to(self.device)

        dataset_x = TabularDataset(x)
        dataloader = DataLoader(dataset_x, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for batch in dataloader:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)
                outputs,_ = self(batch)
                loss = self.criterion(outputs,batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch.size(0)

            epoch_loss = epoch_loss / len(dataset_x)

        with torch.no_grad():
            self.eval()
            _,hidden_representation = self(x)

        return hidden_representation


class TabularDataset(Dataset):
    def __init__(self, features):
        self.features = features  # already tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]






