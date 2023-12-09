import spacy as sp
import benepar as bp
from tqdm import tqdm
import re
from load_conversations import load_json
from collections import Counter
from random import shuffle
#setup
bp.download("benepar_en3_large")

pipe=sp.load("en_core_web_md")

if sp.__version__.startswith("2"):
    pipe.add_pipe(bp.BeneparComponent("benepar_en3"))
else:
    pipe.add_pipe("benepar", config={"model": "benepar_en3"})

convos=load_json("./ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json")
print(type((convos)))
shuffle(convos)
nouns=Counter()
for sents in tqdm(convos[:500]):
    if len(sents) > 512:
        continue
    else:
        doc = pipe(sents)


        if len(list(doc.sents))>0:
            sents=list(doc.sents)[0]
            

            parse_tree=[sent._.parse_string for sent in sents]

            for word in parse_tree:
                if "VB"  in word:
                    nouns[word]+=1

print(nouns.most_common(10))


