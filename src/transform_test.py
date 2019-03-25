import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from gensim.models import KeyedVectors

from argparse import ArgumentParser


class VectorTransformer(nn.Module):
    def __init__(self, source_dim, target_dim):
        super(VectorTransformer, self).__init__()
        self.lmat = nn.Linear(source_dim, target_dim)

    def forward(self, x):
        out = self.lmat(x)
        return out


def create_loader(vocab, source_emb, target_emb):
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
    x = torch.tensor(x, dtype=torch.float32).to("cpu")
    y = torch.tensor(y, dtype=torch.float32).to("cpu")
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    return loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_embedding")
    parser.add_argument("--target_embedding")

    args = parser.parse_args()
    binary = False

    if "bin" in args.source_embedding:
        binary = True

    print("Loading source embedding...")
    source = KeyedVectors.load_word2vec_format(
        args.source_embedding, binary=binary)
    binary = False

    if "bin" in args.target_embedding:
        binary = True

    print("Loading target embedding...")
    target = KeyedVectors.load_word2vec_format(
        args.target_embedding, binary=binary)

    source_words = set(source.vocab.keys())
    target_words = set(target.vocab.keys())

    intersection = source_words.intersection(target_words)
    print(f"Intersection words: {len(intersection)}")

    print("Creating loader...")
    loader = create_loader(intersection, source, target)

    model = VectorTransformer(source.vector_size, target.vector_size)
    model.to("cpu")
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    print("Training...")
    for _ in range(3):
        model.train()
        avg_loss = 0.
        for (x_batch, y_batch) in loader:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(loader)
        print(f"avg_loss: {avg_loss:.3f}")

    source_only_words = source_words - intersection

    for _ in range(10):
        word = source_only_words.pop()
        emb = source.get_vector(word)
        tensor = torch.tensor(emb, dtype=torch.float32).to("cpu")
        pred = model(tensor).detach().numpy()
        similar = target.similar_by_vector(pred)
        print(f"word: {word}")
        print(f"similar: {similar}")
