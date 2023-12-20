import torch
from sentence_transformers import SentenceTransformer,util
import numpy as np
import pandas as pd

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




if __name__ == '__main__':
    embeddings = make_embeddings(ds['Prompt'])
    print(embeddings)




