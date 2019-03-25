import os
import random

import torch

import numpy as np

from pathlib import Path
from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_sequence(train, test, max_features=95000, maxlen=150):
    tk = Tokenizer(lower=True, filters="", num_words=max_features)
    full_text = list(train.values) + list(test.values)
    tk.fit_on_texts(full_text)

    train_tokenized = tk.texts_to_sequences(train.fillna("missing"))
    test_tokenized = tk.texts_to_sequences(test.fillna("missing"))

    X_train = pad_sequences(train_tokenized, maxlen=maxlen)
    X_test = pad_sequences(test_tokenized, maxlen=maxlen)
    return X_train, X_test, tk.word_index


def prepare_emb(word_index, base_w2v, expanded_w2v, max_features=95000):
    all_embs = np.stack(base_w2v.vectors)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    n_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std,
                                        (n_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        try:
            embedding_vector = base_w2v.get_vector(word)
            embedding_matrix[i] = embedding_vector
        except KeyError:
            try:
                embedding_vector = expanded_w2v[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                pass
    return embedding_matrix


def load_w2v(word_index, filepath, max_features=95000):
    path = Path(filepath)
    assert path.exists()
    assert path.is_file()
    if ".bin" in path.name:
        binary = True
    else:
        binary = False
    embeddings_dict = KeyedVectors.load_word2vec_format(path, binary=binary)
    all_embs = np.stack(embeddings_dict.vectors)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    n_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std,
                                        (n_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        try:
            embedding_vector = embeddings_dict.get_vector(word)
            embedding_matrix[i] = embedding_vector
        except KeyError:
            pass
    return embedding_matrix