import torch
from sentence_transformers import SentenceTransformer,util
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
i=1
ds_name=f"out/VERB/VERB_wQuestions{3}.jsonl"

ds=pd.read_json(ds_name,orient="records",lines=True)

def make_embeddings(to_embed:np.array,model:str='paraphrase-albert-small-v2',normalize=True,device:str='cuda')->np.array:
    embedding_model=SentenceTransformer(model,device=device)
    return embedding_model.encode(
        to_embed,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=normalize
    )

def inner_prod_sim(q,a):
    return abs(np.dot(q,a))

def cosine_sim(q,a):
    return abs(np.dot(q,a)/np.sqrt(a.dot(a))*np.sqrt(q.dot(q)))

if __name__ == '__main__':
    answers = make_embeddings(ds['Prompt'])
    questions=make_embeddings(np.unique(ds['Type']))
    
    print(questions.shape)
    print(inner_prod_sim(questions[0],answers[0]))

    vector_store = deque()
    for q in tqdm(range(questions.shape[0])): 
        sims=deque()
        
        for a in range(answers.shape[0]):
            sims.append(cosine_sim(questions[q],answers[a]))
        
        vector_store.append((ds['Type'][q],list(zip(ds['Prompt'],np.array(sims)))  ))

    print([vec for vec in vector_store[1]])




