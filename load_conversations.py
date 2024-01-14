from pathlib import Path
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import deque
import re
from random import sample
from multiprocessing import Queue, Pool, set_start_method



#loads every questions and answer
def load_json(path:Path)-> list[str]:
    if os.path.exists(str(path)):
        ds=pd.read_json(path)
        convos=ds['conversations']
        sents=deque()
        for qa in tqdm(convos):
            for sent in qa:
                sents.append(sent['value'].lower())
        return list(sents)
    return []

#only the first question in the conversation json object
def load_initial_questions(path:Path)->list[str]:
    if os.path.exists(str(path)):
        ds=pd.read_json(path)
        convos=ds['conversations']
        qs=deque()
        for qa in tqdm(convos):
            if len(qa)>0: qs.append(qa[0]['value'].lower())
        return list(qs)
    return []


#loads first x questions from dataset
def load_first_x(path:Path, x:int=1):
    if os.path.exists(str(path)):
        ds=pd.read_json(path)
        convos=ds['conversations']
        qs=deque()
        m=""
        for qa in tqdm(convos):
            for i,sent in enumerate(qa):
                if sent['from'] == 'human' and i<x: 
                    m+=sent['value']
            qs.append(m)
        return list(qs)
    return []


#loads every question tagged human
def load_questions(path:Path)->list[str]:
    if os.path.exists(str(path)):
        ds=pd.read_json(path)
        convos=ds['conversations']
        qs=deque()
        for qa in tqdm(convos):
            for sent in qa:
                if sent['from'] == 'human': 
                    qs.append(sent['value'])
        return list(qs)
    return []


#transfom every question into a json record   
def xlt_questions(path:Path,fname:str)-> None:
    if os.path.exists(str(path)):
        ds=pd.read_json(path)
        convos=ds['conversations']
        qs=deque()

        
        with open(fname+'.jsonl',encoding='utf-8',mode='a') as f:
            
            for qa in tqdm(convos):
                for sent in qa:
                    if sent['from'] == 'human': 
                        f.write(pd.DataFrame([sent['value']], columns=['questions']).to_json(orient='records',lines=True))
        print("Wrote data to "+fname+'.jsonl')

#MOSS3 


#only english or chinese
def check_en(q:str)->False:
    try:
        q.encode(encoding="ascii", errors='strict')
    except UnicodeEncodeError:
        return False
    return True


#TODO subtract prefix and suffix
def get_question(chat:str)->str:
    end = re.compile(r'<eo[ah]>')
    #get human question
    return re.split(end,chat)[0]

#return top x questions as a string
def get_x_questions(convo:dict,x:int)->deque:

    if x>len(convo.keys()): return None
    questions=deque()
    end = re.compile(r"<\|Human\|>:|<eo[ah]>")
    for i in range(x):
        turn = f"turn_{i+1}"
        questions.append(re.sub(end,'',convo[turn]['Human']))
    return questions


def get_x_question(convo:dict,x:int)->str:
    if x> len(convo.keys()): return None
    end = re.compile(r"<\|Human\|>:|<eo[ah]>")
    return re.sub(end,'',convo[f'turn_{x}']['Human'])
    

#tranfoms_moss Dataset to json records form that we care about
def xlt_MOSS3_FirstQuestion(path:Path,fname:str)->None:
    if os.path.exists(str(path)):
        ds=pd.read_json(path,lines=True,orient='records', chunksize=5)
       
        file = fname+'.jsonl'
        
        with open(file,encoding='utf-8',mode='a') as f:
            for chunk in tqdm(ds):
                
                for item in chunk['chat']:
                    category = list(chunk['category'])[0]
                    if check_en(item["turn_1"]['Human']):
                        f.write(
                            pd.DataFrame(
                                [[
                                    category,
                                    get_x_questions(item,1)
                                ]],
                                columns=['category','questions']
                            ).to_json(orient='records', lines=True)
                        )
        print(f"wrote to {file} in json records format")

def make_line(category:str,questions:str,l:bool):
    if l : return [category,questions]
    else: return {'category':category, 'questions':questions} 

def subsample_dataset(path:Path,fname:str,nrows:int,x:int=1):
    if os.path.exists(str(path)):
        ds=pd.read_json(path,lines=True,orient='records', chunksize=50, nrows=nrows)
       
        file = f"{fname}_NQs_{x}.jsonl"
        
        sents=deque()
        index =0
        for chunk in tqdm(ds):            
            for item in chunk['chat']:
                category = list(chunk['category'])[0]
                if check_en(item["turn_1"]['Human']):
                        sents.append([category,get_x_questions(item,x)])

        print("Writing obj.....")
        with open(file,encoding='utf-8',mode='a') as f:

            f.write(
                pd.DataFrame(
                    sents,
                    columns=['category','questions']
                ).to_json(orient='records', lines=True)
            )
        print(f"wrote to {file} in json records format of the first {x} questions")

def random_subsample_dataset(path:Path,fname:str,nrand:int,chunksize:int=50,x:int=1):
    assert nrand < chunksize and "# Unique samples have to be less than chunk size"

    if os.path.exists(str(path)):
        #for minimizing memory footprint returns iterator
        ds=pd.read_json(path,lines=True,orient='records', chunksize=chunksize)
        
        #Given by File name to write
        file = f"{fname}_NQs_{x}.jsonl"
        #choose nrand indicies that we will ignore out of chunksize
        rands = dict.fromkeys(sample(list(range(50)),nrand))
        sents=deque()
        index =0
        for chunk in tqdm(ds):            
            for i,item in enumerate(chunk['chat']):
                if i in rands:
                    category = list(chunk['category'])[0]
                    if check_en(item["turn_1"]['Human']):
                            q=get_x_question(item,x)
                            if q is not None:
                                #append category of question and the x (1,2,3 ... ) question
                                sents.append([category,q])
        
        print("Writing obj.....")
        with open(file,encoding='utf-8',mode='w') as f:

            f.write(
                pd.DataFrame(
                    sents,
                    columns=['category','questions']
                ).to_json(orient='records', lines=True)
            )
        print(f"wrote to {file} in json records format of the first {x} questions")
        return
    print("path does not exist")

def add_q(q,category:str,item,x:int):
    if check_en(item["turn_1"]['Human']):
        q.put([category,get_x_questions(item,x)])


                
                   
                    

                    


   
#tranfoms_moss Dataset to json records form that we care about
def xlt_MOSS3_x_questions(path:Path,fname:str,x:int)->None:
    if os.path.exists(str(path)):
        ds=pd.read_json(path,lines=True,orient='records', chunksize=50)
       
        file = f"{fname}_NQs_{x}.jsonl"
        
        sents=deque()
        index =0
        for chunk in tqdm(ds): 
            category = list(chunk['category'])[0]           
            for item in chunk['chat']:
                
                if check_en(item["turn_1"]['Human']):
                        q=get_x_questions(item,x)
                        
                        if  q!=None: 
                            
                            q.appendleft(category)
  
                            sents.append(list(q))
                       
            
                
        print("Writing obj.....")
        with open(file,encoding='utf-8',mode='a') as f:

            f.write(
                pd.DataFrame(
                    list(sents),
                    columns=['category']+[f'question_{i}' for i in range(x)]
                ).to_json(orient='records', lines=True)
            )
        print(f"wrote to {file} in json records format of the first {x} questions")
        
 #tranfoms_moss Dataset to json records form that we care about
def xlt_MOSS3_x_question(path:Path,fname:str,x:int)->None:
    if os.path.exists(str(path)):
        ds=pd.read_json(path,lines=True,orient='records', chunksize=50)
       
        file = f"{fname}_QN_{x}.jsonl"
        
        sents=deque()
        index =0
        for chunk in tqdm(ds): 
            category = list(chunk['category'])[0]           
            for item in chunk['chat']:
                
                if check_en(item["turn_1"]['Human']):
                        q=get_x_question(item,x)
                        if q != None: sents.append([category,q])
            
                
        print("Writing obj.....")
        with open(file,encoding='utf-8',mode='a') as f:

            f.write(
                pd.DataFrame(
                    sents,
                    columns=['category','questions']
                ).to_json(orient='records', lines=True)
            )
        print(f"wrote to {file} in json records format of the first {x} questions")           

def xlt_all(path:Path)-> None:
    if os.path.exists(str(path)):
        ds=pd.read_json(path)
        convos=ds['conversations']
        sents=deque()
        for qa in tqdm(convos):
            for sent in qa:
                sents.append(sent['value'].lower())
        return list(sents)
    return []

if __name__ == "__main__":

    # # print(xlt_questions("./ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json","xlt_questions"))
    for i in range(10):
        xlt_MOSS3_x_questions("../moss-003-sft-no-tools/moss-003-sft-no-tools.jsonl",'rand_Moss3_XLT',x=i+1)


   


