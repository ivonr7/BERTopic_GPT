import numpy as np

import spacy as sp
from tqdm import tqdm
from collections import Counter, deque
import click

import pandas as pd
from collections import deque
import os
from itertools import product

#splits sentences into verbs Words well on large data will be slow on small data
def get_verbs(col:pd.Series=None,pos:str='VERB') -> pd.DataFrame:
    if col is None: return None
    #Category aggregator deque for fast insert speeds
    grammar_cats=deque()

    #Copy all questions to gpu and tag verbs
    pipe = sp.load("en_core_web_md")
    #preccess all questions
    sents = pipe.pipe(col)
    
    #get index, verb , sentence for filtering later
    for i,sent in tqdm(enumerate(list(sents))):

        for token in sent:
            #Token is essentially a word-> 
            #if words part of speech (.pos_) is VERB than save it else discard
            if token.pos_ == pos:
                grammar_cats.append([i,token.lemma_,str(sent)])
                
    
    return pd.DataFrame(
        grammar_cats,
        columns=["index","verb","sent"]
    )


def get_Question(question,pos,filename,filter_num=5):

    if filename == '':
        filename =  f'rand_Moss3_XLT_NQs_{question}.jsonl'
    OUT_PATH= os.getcwd()+f'/out/{pos}/'
    OUT_FILE=pos+f"_wQuestions{question}.jsonl"

    

    #load dataset
    print(f"loading dataset {filename}.......")
    df = pd.read_json(filename,orient='records',lines=True)
    
    n_convos  = df.shape[0]

    #make output directory
    if not os.path.exists(OUT_PATH): os.mkdir(OUT_PATH)

    with open(OUT_PATH+OUT_FILE, mode="w") as f:
        print("file created")


    print("Extracting Verbs:---------------------------------------------------")
    with open(OUT_PATH+OUT_FILE, mode="a") as f:


        verb_map = get_verbs(df[f'question_{question-1}'])
        print("Filtering---------------------------------------------------")
        #filter fisrt top x
        verb_map = filter_df(verb_map,'verb',filter_num)

        f.write(
            verb_map.to_json(orient='records',lines=True)
        )   
    #filter whole dataset on top_x verbs
    out = df.iloc[np.unique(verb_map['index'])].copy()

    #add back all verbs using their row index
    #multiples is handle by using a list
    verbs = deque()
    for i in tqdm(range(df.shape[0])):
        v=verb_map['verb'].where(i == verb_map['index']).dropna()
        if v.shape[0] != 0: verbs.append(list(v))
    
    out.insert(1,f'Verb_{question}',list(verbs))
    out.to_json(f"out/VERB/filterdataset_{question}.jsonl", orient='records',lines=True)

    print("Statistics---------------------------------------------------")
    agg_verbs = np.array(verb_map['verb']).flatten()
    # #print common verbs



    commons=Counter(agg_verbs)
    total_v = sum([commons[key] for key in commons.keys()])
    print(total_v)
    print([(common[0],  
            f"Percentage in respect to all verbs: {common[1]/n_convos:.3f}",
            f"{common[1]} occurences") 
        for common in commons.most_common(filter_num)])
    print(f"{total_v} total verbs {out.shape[0]} filtered conversations {n_convos} total conversations")


    # print(np.unique(grammars))

def get_max_of_list(l:list[list]):
    return max([len(item) for item in l])

#filters entire dataset into component verbs


def get_dataset(question,pos,filename,filtering=True):

    if filename == '':
        filename =  f'rand_Moss3_XLT_NQs_{question}.jsonl'
    OUT_PATH= os.getcwd()+f'/out/{pos}/'
    OUT_FILE=pos+f"_wQuestions{question}.jsonl"

    pipe = sp.load("en_core_web_md")

    #load dataset
    print(f"loading dataset {filename}.......")
    df = pd.read_json(filename,orient='records',lines=True)
    #Category aggregator deque for fast insert speeds
    grammar_cats=deque()


    #make output directory
    if not os.path.exists(OUT_PATH): os.mkdir(OUT_PATH)

    with open(OUT_PATH+OUT_FILE, mode="w") as f:
        print("file created")


    print("creating dataset...")
    with open(OUT_PATH+OUT_FILE, mode="a") as f:
        
        #create all sents
        cols=[]
        for i in range(question):
            cols.append(pipe.pipe(df[f'question_{i}']))

        verb_cols=[]
        grammar_cats=[]
        #get all verbs
        for i,col in enumerate(cols):
            print(f"Getting Verbs Col {i+1}")
            for i,sent in tqdm(enumerate(list(col))):

                for token in sent:
                    if token.pos_ == pos:
                        row = list(df.iloc[i])
                        grammar_cats.append((i,token.lemma_)) #make list of tuples and convert to dataframe for indexing
            verb_cols.append(pd.DataFrame(grammar_cats,columns=['index','verb'])) 
            
        del cols
        del grammar_cats
        
        #index each col dataframe with index to get 10 series
        #series->list 
        #cartesian product of all lists
        #append all outpus of cartesian product
        rows=[]
        for i in tqdm(range(df.shape[0])): #iterate rows
            axes=[]
            for col in verb_cols:
                
                axes.append(list(col['verb'].where(col['index']==i).dropna())) 
            m=get_max_of_list(axes)

            for i in range(len(axes)):
                for j in range(m-len(axes[i])):
                    axes[i].append(np.nan)
            a=pd.DataFrame(
                    axes
                ).ffill(axis=1).T.copy()
            rows.append(
                a.to_dict(orient='records')
            )

        print(rows[0])

            
            
            

        print(rows[0])
        exit(1)  
        cols = df.columns
        out = pd.DataFrame(
            grammar_cats,
            columns=[*cols,"Verb"]
        )   
        



        print("Validating")
        top_x = get_top_x(out['Verb'],5)
        passed = True
        for row in range(out.shape[0]):
                if out.iloc[row]['Verb'] not in top_x: passed=False
        print("passed" if passed else "failed")

        print(f"writing {pos} and prompts to {OUT_FILE}.......")
        
        f.write(
            out.to_json(orient='records',lines=True)
        )


        
    


#for shit

def to_verbs(col:pd.Series,pipe,pos='VERB'):
        return pipe.pipe(col)


def set_verb(out_key:str,col,pipe,pos='VERB')->tuple:
    sents = pipe.pipe(col)
    lines=deque()    
    for i,sent in tqdm(enumerate(list(sents))):
  

        for token in sent:
            if token.pos_ == pos:
                lines.append((i,token.lemma_))
    #return as dict
    return out_key,list(lines)

def create_verbs(df:pd.DataFrame,pipe)->pd.DataFrame:
    n_cols = dict()
    for col in df.columns:
            print(f"Running {col}")
            if not col == 'category':
                print("copying to gpu...")
                if col == 'question_0':
                    c = set_verb('category',df[col],pipe)
                    n_cols[c[0]] = c[1]
                    n_cols['question_0']=list(df['question_0']) 
                    
                else: 
                    c = set_verb(col,df[col],pipe)
                    n_cols[c[0]] = c[1]


    lines = itertools.product(
        n_cols # iterable unpacking
    )
    return pd.DataFrame(lines)
def filter_by_verbs_in_top_x(df:pd.DataFrame,col_key:str,verbs:pd.Series,top_x:int):
    df=df[df[col_key].isin(get_top_x(verbs,top_x))]   


def filter_df(df:pd.DataFrame,col_key:str,top_x:int):
    return df[df[col_key].isin(get_top_x(df[col_key],top_x))].copy()

def get_top_x(col:pd.Series,x:int)->list:
    return [obj[0] for obj in (Counter(list(col)).most_common(x))]

def get_Convo_Trace(question,pos,filename):

    if filename == '':
        filename =  f'rand_Moss3_XLT_NQs_{question}.jsonl'
    OUT_PATH= os.getcwd()+f'/out/{pos}/'
    OUT_FILE=pos+f"_wTop5q_convos{question}.jsonl"

    pipe = sp.load("en_core_web_md")

    #load dataset
    print(f"loading dataset {filename}.......")
    df = pd.read_json(filename,orient='records',lines=True)
    #Category aggregator deque for fast insert speeds
    


    #make output directory
    if not os.path.exists(OUT_PATH): os.mkdir(OUT_PATH)

    with open(OUT_PATH+OUT_FILE, mode="w") as f:
        print("file created")
    


    with open(OUT_PATH+OUT_FILE, mode="a") as f:

        transform_df=create_verbs(df,pipe)
        print(transform_df.head)
        for col in transform_df.columns:
            if not col == 'question_0' : 
                if col == 'question_1':filter_df(transform_df,col,5)
                else: filter_df(transform_df,col,20)

        f.write(
            transform_df.to_json(orient='records',lines=True)
        )
        

        
            


        


        
    

if __name__ == '__main__':
    
    for i in range(10):
        get_Question(i+1,'VERB',f'rand_Moss3_XLT_NQs_{i+1}.jsonl',filter_num=20)
    # N_QUESTION = 1
    # CATEGORY="VERB"
    # OUT_PATH= os.getcwd()+f'/out/{CATEGORY}/'
    # ds_name = f'rand_Moss3_XLT_NQs_{N_QUESTION}.jsonl'


    # OUT_FILE=CATEGORY+f"_wQuestions{N_QUESTION}.jsonl"




    # pipe = sp.load("en_core_web_md")


    # #load dataset
    # print(f"loading dataset {ds_name}.......")
    # df = pd.read_json(ds_name,orient='records',lines=True, chunksize=10000)




    # #Category aggregator deque for fast insert speeds
    # grammar_cats=deque()


    # #make output directory
    # if not os.path.exists(OUT_PATH): os.mkdir(OUT_PATH)

    # with open(OUT_PATH+OUT_FILE, mode="w") as f:
    #     print("file created")
    # total_q=0


    # with open(OUT_PATH+OUT_FILE, mode="a") as f:
    #     for chunk in tqdm(df):

    #         doc = list(chunk['questions'])
    #         sents = pipe.pipe(doc)
            
    #         for sent in tqdm(list(sents)):
    #             total_q+=1
    #             for token in sent:
    #                 if token.pos_ == CATEGORY:
    #                     grammar_cats.append([token.lemma_,str(sent)])
                        
    #     print(f"writing {CATEGORY} and prompts to {OUT_FILE}.......")
    #     f.write(
    #         pd.DataFrame(
    #             grammar_cats,
    #             columns=["Type","Prompt"]
    #         ).to_json(orient='records',lines=True)
    #     )


        
    # grammars=[item[0] for item in grammar_cats]

    # #print common verbs
    # commons=Counter(grammars)
    # total_v = sum([commons[key] for key in commons.keys()])
    # print(total_v)
    # print([(common[0],  
    #         f"Percentage in respect to all verbs: {common[1]/total_v:.3f}",
    #         f"Pecentage in respect to all questions:{common[1]/total_q:.3f} "
    #         ) 
    #     for common in commons.most_common(20)])


    # print(np.unique(grammars))




