import torch
import spacy

if torch.cuda.is_available():
    bt=torch.device("cuda")
else:
    bt=torch.device("cpu")

sp=spacy.require_gpu()

print("BERTopic Using ",bt)
print("Spacy using","gpu" if sp else "cpu")




# Return to University/College
# Masters of Management/Marketing
# Networking: attend career fairs,  connect with previous contacts for finding Director of Marketing/Management position