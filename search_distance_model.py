import torch
from sentence_transformers import SentenceTransformer,util
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
i=1
ds_name=f"out/VERB/VERB_wQuestions{3}.jsonl"

ds=pd.read_json(ds_name,orient="records",lines=True)

def make_embeddings(to_embed:np.array)->np.array:
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2',device='cuda')
    return embedding_model.encode(
        to_embed,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

def inner_prod_sim(q,a):
    return abs(np.dot(q,a))


if __name__ == '__main__':
    answers = make_embeddings(ds['Prompt'])
    questions=make_embeddings(np.unique(ds['Type']))
    print(questions.shape)
    print(inner_prod_sim(questions[0],answers[0]))

    vector_store = deque()
    for q in tqdm(range(questions.shape[0])): 
        sims=deque()
        
        for a in range(answers.shape[0]):
            sims.append(inner_prod_sim(questions[q],answers[a]))
        
        vector_store.append(np.array(sims))

    print(vector_store[0:5])




