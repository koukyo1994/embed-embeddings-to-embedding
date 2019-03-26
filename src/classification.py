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
    from util import (timer, get_logger, check_format,
                      count_words_in_expanded_words)
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
        source = KeyedVectors.load_word2vec_format(
            args.source_embedding, binary=check_format(args.source_embedding))
        target = KeyedVectors.load_word2vec_format(
            args.target_embedding, binary=check_format(args.target_embedding))
        expanded_emb = embedding_expander(source, target, logger)
        embedding_matrix_expanded, in_base, in_expanded, expanded_words = \
            prepare_emb(word_index, target, expanded_emb, 95000)
        logger.info(f"In base embedding: {in_base}")
        logger.info(f"In expanded embedding: {in_expanded}")

        embedding_matrix, in_base = load_w2v(word_index, args.target_embedding,
                                             95000)
        logger.info(f"In base embedding: {in_base}")

    # check the number of words in expanded words list
    expanded_words = set(expanded_words)
    num_words = train["tokenized"].map(
        lambda x: count_words_in_expanded_words(x, expanded_words))
    num_words_test = test["tokenized"].map(
        lambda x: count_words_in_expanded_words(x, expanded_words))
    nrow = num_words[num_words != 0].shape[0]
    nrow_test = num_words_test[num_words_test != 0].shape[0]

    nwords_max = num_words.max()
    nwords_max_test = num_words_test.max()

    nwords_mean = num_words.mean()
    nwords_mean_test = num_words_test.mean()

    logger.info(f"Number of train rows contains expanded words: {nrow}")
    logger.info(
        f"Ratio of rows in train containig expanded words: {nrow / X.shape[0]}"
    )
    logger.info(f"Number of test rows contains expanded words: {nrow_test}")
    logger.info(
        f"Ratio of rows in test containing expanded words: {nrow_test / X_test.shape[0]}"
    )
    logger.info(f"Max expanded words in train: {nwords_max}")
    logger.info(f"Max expanded words in test: {nwords_max_test}")
    logger.info(f"Mean expanded words in train: {nwords_mean}")
    logger.info(f"Mean expanded words in test: {nwords_mean_test}")

    # only fastText
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
            "n_attention": 50
        })
    trainer.fit(X, y.values, 30)

    # with Expanded Words
    trainer_ex = NNTrainer(
        NeuralNet,
        logger,
        device=args.device,
        kwargs={
            "embedding_matrix": embedding_matrix_expanded,
            "n_classes": 9,
            "hidden_size": 64,
            "maxlen": 1200,
            "linear_size": 100,
            "n_attention": 50
        })
    trainer_ex.fit(X, y.values, 20)

    path = Path(f"figure/{trainer.tag}")
    path.mkdir(parents=True, exist_ok=True)

    idx = np.arange(len(trainer.f1s[0]))
    for i in range(trainer.n_splits):
        plt.figure()
        plt.plot(idx, trainer.scores[i], label="Accuracy")
        plt.plot(idx, trainer_ex.scores[i], label="Accuracy expanded")
        plt.plot(idx, trainer.f1s[i], label="F1")
        plt.plot(idx, trainer_ex.f1s[i], label="F1 expanded")
        plt.legend()
        plt.savefig(path / f"score{i}.png")

        plt.figure()
        plt.plot(idx, trainer.loss[i], label="Train Loss")
        plt.plot(idx, trainer_ex.loss[i], label="Train Loss expanded")
        plt.plot(idx, trainer.loss_val[i], label="Validation Loss")
        plt.plot(idx, trainer_ex.loss_val[i], label="Validation Loss expanded")
        plt.legend()
        plt.savefig(path / f"Loss{i}.png")

    test_preds = trainer.predict(X_test)
    score = accuracy_score(y_test.values, np.argmax(test_preds, axis=1))
    f1 = f1_score(
        y_test.values, np.argmax(test_preds, axis=1), average="macro")
    logger.info(f"Test Acc: {score}")
    logger.info(f"Test f1: {f1}")

    test_preds = trainer_ex.predict(X_test)
    score = accuracy_score(y_test.values, np.argmax(test_preds, axis=1))
    f1 = f1_score(
        y_test.values, np.argmax(test_preds, axis=1), average="macro")
    logger.info(f"Test Acc expanded: {score}")
    logger.info(f"Test f1 expanded: {f1}")