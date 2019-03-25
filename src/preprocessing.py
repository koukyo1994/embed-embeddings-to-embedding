import re
import MeCab
import mojimoji

import subprocess

proc_returns = subprocess.run(["mecab-config", "--dicdir"],
                              capture_output=True)
dicdir = proc_returns.stdout.decode("utf8").replace("\n", "")

tagger = MeCab.Tagger(f"-Ochasen -d {dicdir}/mecab-ipadic-neologd")


def tokenizer(text):
    text = mojimoji.zen_to_han(text.replace("\n", ""), kana=False)
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    parsed = tagger.parse(text).split("\n")
    parsed = [t.split("\t") for t in parsed]
    parsed = list(filter(lambda x: x[0] != "" and x[0] != "EOS", parsed))
    parsed = [p[2] for p in parsed]
    return parsed
