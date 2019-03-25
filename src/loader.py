import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, path):
        path = Path(path)
        assert path.exists()
        assert path.is_dir()

        self.classes = list()
        self.dir = list()
        for p in path.iterdir():
            if p.is_dir():
                self.classes.append(p.name)
                self.dir.append(p)

        self.encode_dict = {k: i for i, k in enumerate(self.classes)}
        self.decode_dict = {i: k for i, k in enumerate(self.classes)}

        self.title = list()
        self.url = list()
        self.datetime = list()
        self.text = list()
        self.label = list()

        for p in self.dir:
            for d in p.iterdir():
                if d.name != "LICENSE.txt":
                    with open(d) as f:
                        split = f.readlines()
                    self.url.append(split[0].replace("\n", ""))
                    self.datetime.append(split[1].replace("\n", ""))
                    self.title.append(split[2].replace("\n", ""))
                    self.text.append(" ".join([
                        s.replace("\n", "").replace("\u3000", " ")
                        for s in split[3:]
                    ]))
                    self.label.append(self.encode_dict[p.name])
        self.data = pd.DataFrame({
            "date": self.datetime,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "label": self.label
        })

        self.tokenized = False

    def tokenize(self, tokenizer, kwargs={}):
        self.data["concatenated"] = self.data["title"] + " " + \
            self.data["text"]
        self.data["tokenized"] = self.data.concatenated.map(
            lambda x: tokenizer(x, **kwargs))
        self.tokenized = True

    def load(self):
        assert self.tokenized, "Needs tokenization before loading"
        train, test = train_test_split(
            self.data, test_size=0.3, shuffle=True, random_state=42)
        return train, test
