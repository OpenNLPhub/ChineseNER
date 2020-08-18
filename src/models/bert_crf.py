import torch
import os
from torch import nn
from transformers import BertForTokenClassification,BertTokenizer,BertConfig;

cwd=os.getcwd()

class BERT_CRF(nn.Module):
    def __init__(self,vocab_size):
        pass


 