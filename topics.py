from bertopic import BERTopic    
import numpy as np
import pandas as pd
from pathlib import Path
from json import loads
ds_name=""

ds=pd.read_json(ds_name,orient="records",lines=True)



#make topic model for clustering
topic_model=BERTopic(
    language="english",calculate_probabilities=False,
    verbose=True,top_n_words=8,n_gram_range=(1,1),low_memory=True
)

topic_model.fit(ds["Prompt"],y=ds["Type"])

labels=topic_model.generate_topic_labels(topic_prefix=False,separator=", ")
topic_model.set_topic_labels(topic_labels=labels)
print(labels)
print(len(labels))


#VISualizations


#vizualize hierarchy
h_topics=topic_model.hierarchical_topics(cats)
fig = topic_model.visualize_hierarchy(hierarchical_topics=h_topics)
fig.write_html("./viz/viz_hier.html")

#visualize keyword frequency
fig= topic_model.visualize_barchart()
fig.write_html(f"./viz/viz_bar{i}.html")

#vizualize clusters
fig = topic_model.visualize_documents(cats)
fig.write_html(f"./viz/viz_docs{i}.html")

#vizualize similarity
fig = topic_model.visualize_heatmap()
fig.write_html(f"./viz/viz_heatmap{i}.html")




