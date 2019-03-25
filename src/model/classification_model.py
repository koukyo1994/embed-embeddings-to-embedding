import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_dims, step_dims, n_middle, n_attention,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.support_masking = True
        self.feature_dims = feature_dims
        self.step_dims = step_dims
        self.n_middle = n_middle
        self.n_attention = n_attention
        self.features_dim = 0

        self.lin1 = nn.Linear(feature_dims, n_middle, bias=False)
        self.lin2 = nn.Linear(n_middle, n_attention, bias=False)

    def forward(self, x, mask=None):
        step_dims = self.step_dims

        eij = self.lin1(x)
        eij = torch.tanh(eij)
        eij = self.lin2(eij)

        a = torch.exp(eij).reshape(-1, self.n_attention, step_dims)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 2, keepdim=True) + 1e-10

        weighted_input = torch.bmm(a, x)
        return torch.sum(weighted_input, 1)


class NeuralNet(nn.Module):
    def __init__(self,
                 embedding_matrix,
                 n_classes=9,
                 hidden_size=64,
                 maxlen=150,
                 linear_size=100,
                 n_attention=30):
        super(NeuralNet, self).__init__()

        n_features, embed_size = embedding_matrix.shape
        self.embedding = nn.Embedding(n_features, embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.2)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(
            hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.attn = Attention(hidden_size * 2, maxlen, n_attention,
                              n_attention)
        self.linear = nn.Linear(hidden_size * 2, linear_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(linear_size, n_classes)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        h_attn = self.attn(h_gru)
        linear = self.relu(self.linear(h_attn))
        h_drop = self.dropout(linear)
        out = self.linear2(h_drop)

        return out
