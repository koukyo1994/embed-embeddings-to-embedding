import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class VectorTransformer(nn.Module):
    def __init__(self, source_dim, target_dim):
        super(VectorTransformer, self).__init__()
        self.lmat = nn.Linear(source_dim, target_dim)

    def forward(self, x):
        out = self.lmat(x)
        return out


class RandomVectorTransformer:
    def __init__(self, input_dim, output_dim, loss_fn=nn.MSELoss):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_fn = loss_fn
        self.models = list()

    def fit(self, dataloaders, device="cpu", n_epochs=30):
        for dataloader in dataloaders:
            model = VectorTransformer(self.input_dim, self.output_dim)
            model.to(device)
            optimizer = optim.Adam(model.parameters())
            loss_fn = self.loss_fn()

            for _ in range(n_epochs):
                model.train()
                avg_loss = 0.
                for (x_batch, y_batch) in dataloader:
                    y_pred = model(x_batch)

                    if loss_fn == nn.MSELoss:
                        dummy = torch.ones((y_batch.size(0), ))
                        loss = loss_fn(y_pred, y_batch, dummy)
                    else:
                        loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(dataloader)
                print(f"avg_loss: {avg_loss:.3f}")
                self.models.append(model)

    def predict(self, x):
        out = np.zeros(self.output_dim)
        n_estimators = len(self.models)
        for model in self.models:
            model.eval()
            out += model(x).detach().cpu().numpy() / n_estimators
        return out


def create_loader(vocab, source_emb, target_emb, device):
    n_vec = len(vocab)
    source_dim = source_emb.vector_size
    target_dim = target_emb.vector_size

    x = np.zeros((n_vec, source_dim))
    y = np.zeros((n_vec, target_dim))
    for i, key in enumerate(vocab):
        source_vec = source_emb.get_vector(key)
        target_vec = target_emb.get_vector(key)
        x[i, :] = source_vec
        y[i, :] = target_vec
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    return loader


def embedding_expander(source, target, logger):
    source_words = set(source.vocab.keys())
    target_words = set(target.vocab.keys())
    intersection = source_words.intersection(target_words)
    logger.info(f"Intersection words: {len(intersection)}")

    logger.info(f"Creating loader...")
    loader = create_loader(intersection, source, target, "cpu")
    model = VectorTransformer(source.vecor_size, target.vector_size)
    model.to("cpu")
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    logger.info(f"Training Vector Transformer...")
    for i in range(3):
        model.train()
        avg_loss = 0.
        for (x_batch, y_batch) in loader:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(loader)
        logger.info(f"Epoch {i + 1} avg_loss: {avg_loss:.4f}")
        source_only_words = source_words - intersection

        expanded_embedding = dict()
        for word in source_only_words:
            emb = source.get_vector(word)
            tensor = torch.tensor(emb, dtype=torch.float32).to("cpu")
            pred = model(tensor).detach().numpy()
            expanded_embedding[word] = pred
        return expanded_embedding