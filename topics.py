from bertopic import BERTopic
from bertopic.representation import PartOfSpeech, ZeroShotClassification
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
i=1
ds_name=f"out/VERB/VERB_wQuestions{3}.jsonl"

ds=pd.read_json(ds_name,orient="records",lines=True)

rep_model = PartOfSpeech(
    "en_core_web_md",
    pos_patterns=[
        {'POS':'VERB'}
    ]
                         
)

#Supervised modeling
empty_dim_red=BaseDimensionalityReduction()
clf=LogisticRegression()
pca= PCA(n_components=5)
kmeans=KMeans(n_clusters=20)
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 1))
ctfidf=ClassTfidfTransformer(
    reduce_frequent_words=True,   
)
zero_shot = ZeroShotClassification(
    [label for label,_ in Counter(ds['Type']).most_common(20)]
)
#make topic model for clustering
topic_model=BERTopic(
    language="english",
    verbose=True,top_n_words=2,n_gram_range=(1,1),low_memory=True,
    nr_topics=100,
    # umap_model=pca,
    # hdbscan_model=kmeans,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf,
    representation_model=zero_shot
)



topic_model.fit_transform(
    ds["Prompt"]#,
    # y=[label for label,_ in Counter(ds['Type']).most_common(20)]
)

labels=topic_model.generate_topic_labels(topic_prefix=False,separator=", ")
topic_model.set_topic_labels(topic_labels=labels)
print(labels)
print(len(labels))
with open("verbs.csv" ,'w') as f:
    for label in labels:
        f.write(label+",\n")

#VISualizations
print("Creating Visualizations")

#vizualize hierarchy
# h_topics=topic_model.hierarchical_topics(ds["Prompt"])
# fig = topic_model.visualize_hierarchy(hierarchical_topics=h_topics)
# fig.write_html("./out/viz/viz_hier.html")

#visualize keyword frequency
fig= topic_model.visualize_barchart()
fig.write_html(f"./out/viz/viz_bar{i}.html")

#vizualize clusters
fig = topic_model.visualize_documents(ds['Prompt'])
fig.write_html(f"./out/viz/viz_docs{i}.html")

#vizualize similarity
fig = topic_model.visualize_heatmap()
fig.write_html(f"./out/viz/viz_heatmap{i}.html")




