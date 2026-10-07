"""Small scikit-learn-style autoencoder used as an optional lens."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


def _activation(name):
    activations = {"ReLU": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"unknown activation function: {name!r}") from exc


def _network(dimension_pairs, activation, batchnorm, dropout, dropout_probability):
    layers = []
    for index, (input_dimension, output_dimension) in enumerate(dimension_pairs):
        layers.append(nn.Linear(input_dimension, output_dimension))
        is_output_layer = index == len(dimension_pairs) - 1
        if not is_output_layer:
            if batchnorm:
                layers.append(nn.BatchNorm1d(output_dimension))
            layers.append(_activation(activation))
            if dropout:
                layers.append(nn.Dropout(dropout_probability))
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layers,
        use_batchnorm,
        use_dropout,
        dropout_prob,
        activation_function,
        learning_rate,
        w_decay,
        batch_size,
        epochs,
        random_state,
    ):
        super().__init__()
        torch.manual_seed(random_state)

        hidden_dimensions = []
        current_dimension = input_dim
        for _ in range(max(0, num_layers - 1)):
            next_dimension = max(2, current_dimension // 2)
            if next_dimension >= current_dimension or next_dimension == 2:
                break
            hidden_dimensions.append(next_dimension)
            current_dimension = next_dimension

        encoder_dimensions = [input_dim, *hidden_dimensions, 2]
        decoder_dimensions = [2, *reversed(hidden_dimensions), input_dim]
        self.encoder = _network(
            list(zip(encoder_dimensions, encoder_dimensions[1:])),
            activation_function,
            use_batchnorm,
            use_dropout,
            dropout_prob,
        )
        self.decoder = _network(
            list(zip(decoder_dimensions, decoder_dimensions[1:])),
            activation_function,
            use_batchnorm,
            use_dropout,
            dropout_prob,
        )

        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_batchnorm = use_batchnorm
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = AdamW(
            self.parameters(), lr=learning_rate, weight_decay=w_decay
        )

    def forward(self, features):
        representation = self.encoder(features)
        reconstruction = self.decoder(representation)
        return reconstruction, representation

    def fit_transform(self, features):
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy(dtype=np.float32, copy=True)
        elif isinstance(features, np.ndarray):
            features = features.astype(np.float32, copy=True)
        else:
            features = features.float()

        features = (
            torch.from_numpy(features)
            if not isinstance(features, torch.Tensor)
            else features
        ).to(self.device)
        if len(features) == 0:
            raise ValueError("the autoencoder cannot fit an empty dataset")
        if self.use_batchnorm and len(features) == 1:
            raise ValueError("batch normalization requires at least two samples")

        batch_size = min(self.batch_size, len(features))
        if self.use_batchnorm and len(features) % batch_size == 1:
            batch_size -= 1
        generator = torch.Generator().manual_seed(self.random_state)
        dataloader = DataLoader(
            TabularDataset(features),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        self.train()
        for _ in range(self.epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                reconstruction, _ = self(batch)
                loss = self.criterion(reconstruction, batch)
                loss.backward()
                self.optimizer.step()

        self.eval()
        with torch.no_grad():
            _, representation = self(features)
        return representation


class TabularDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]
