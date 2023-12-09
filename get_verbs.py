import numpy as np

import spacy as sp
from tqdm import tqdm
from collections import Counter, deque


import pandas as pd

import os

N_QUESTION = 3
CATEGORY="VERB"
OUT_PATH= os.getcwd()+f'/out/{CATEGORY}/'
ds_name = f'rand_Moss3_XLT_NQs_{N_QUESTION}.jsonl'


OUT_FILE=CATEGORY+f"_wQuestions{N_QUESTION}.jsonl"




pipe = sp.load("en_core_web_md")


#load dataset
print(f"loading dataset {ds_name}.......")
df = pd.read_json(str(ds_name),orient='records',lines=True, chunksize=1000)




#Category aggregator deque for fast insert speeds
grammar_cats=deque()


#make output directory
if not os.path.exists(OUT_PATH): os.mkdir(OUT_PATH)

with open(OUT_PATH+OUT_FILE, mode="w") as f:
    print("file created")



with open(OUT_PATH+OUT_FILE, mode="a") as f:
    for chunk in tqdm(df):

        doc = list(chunk['questions'])
        sents = pipe.pipe(doc)
        
        for sent in tqdm(list(sents)):

            for token in sent:
                if token.pos_ == CATEGORY:
                    grammar_cats.append([token.lemma_,str(sent)])
    print(f"writing {CATEGORY} and prompts to {OUT_FILE}.......")
    f.write(
        pd.DataFrame(
            grammar_cats,
            columns=["Type","Prompt"]
        ).to_json(orient='records',lines=True)
    )


    
grammars=[item[0] for item in grammar_cats]

#print common verbs
commons=Counter(grammars)
print(commons.most_common(20))


print(np.unique(grammars))




