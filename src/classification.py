import sys
import argparse

import matplotlib
matplotlib.use("agg")
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("../")
    sys.path.append("./")

    from loader import DataLoader
    from util import timer, get_logger, check_format
    from preprocessing import tokenizer
    from train_helper import to_sequence, prepare_emb, load_w2v
    from model.classification_model import NeuralNet
    from model.vector_transformer import embedding_expander
    from trainer import NNTrainer
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp")
    parser.add_argument("--source_embedding")
    parser.add_argument("--target_embedding")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_epochs", default=10, type=int)

    args = parser.parse_args()
    assert args.source_embedding

    logger = get_logger(name="Main", tag=args.exp)
    with timer("Load Data", logger):
        loader = DataLoader("../input/text")

    with timer("tokenize", logger):
        loader.tokenize(tokenizer)

    train, test = loader.load()
    X = train["tokenized"]
    X_test = test["tokenized"]

    y = train["label"]
    y_test = test["label"]

    with timer("Convert to sequence", logger):
        X, X_test, word_index = to_sequence(
            X, X_test, max_features=95000, maxlen=1200)

    with timer("Load embedding", logger):
        if args.target_embedding:
            source = KeyedVectors.load_word2vec_format(
                args.source_embedding,
                binary=check_format(args.source_embedding))
            target = KeyedVectors.load_word2vec_format(
                args.target_embedding,
                binary=check_format(args.target_embedding))
            expanded_emb = embedding_expander(source, target, logger)
            embedding_matrix, in_base, in_expanded = prepare_emb(
                word_index, target, expanded_emb, 95000)
            logger.info(f"In base embedding: {in_base}")
            logger.info(f"In expanded embedding: {in_expanded}")
        else:
            embedding_matrix, in_base = load_w2v(word_index,
                                                 args.source_embedding, 95000)
            logger.info(f"In base embedding: {in_base}")
    trainer = NNTrainer(
        NeuralNet,
        logger,
        device=args.device,
        kwargs={
            "embedding_matrix": embedding_matrix,
            "n_classes": 9,
            "hidden_size": 64,
            "maxlen": 1200,
            "linear_size": 100,
            "n_attention": 30
        })
    trainer.fit(X, y.values, 30)

    path = Path(f"figure/{trainer.tag}")
    path.mkdir(parents=True, exist_ok=True)

    idx = np.arange(len(trainer.f1s[0]))
    for i in range(trainer.n_splits):
        plt.figure()
        plt.plot(idx, trainer.scores[i], label="Accuracy")
        plt.plot(idx, trainer.f1s[i], label="F1")
        plt.legend()
        plt.savefig(path / f"score{i}.png")

    test_preds = trainer.predict(X_test)
    score = accuracy_score(y_test.values, np.argmax(test_preds, axis=1))
    f1 = f1_score(y_test.values, np.argmax(test_preds, axis=1))
    logger.info(f"Test Acc: {score}")
    logger.info(f"Test f1: {f1}")