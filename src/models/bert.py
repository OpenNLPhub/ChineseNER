import os
import torch
import sys
cwd=os.getcwd()
sys.path.append(os.path.join(cwd,'src'))

from torch import nn
import numpy as np
import torch.optim as optim
from transformers import BertForTokenClassification,BertConfig,BertTokenizer
from progressbar import ProgressBar
from config import TrainingConfig
from utils import tag_add_padding
from copy import deepcopy
progress=ProgressBar()

class bert_chinese_ner(nn.Module):
    model_path=os.path.join(cwd,'data','static','bert-base-chinese')
    def __init__(self,vocab_size,emb_size,hidden_size,num_labels):
        super(bert_chinese_ner,self).__init__() 
        self.bertconfig=BertConfig.from_pretrained(bert_chinese_ner.model_path,num_labels=num_labels,author="lingze")
        self.model=BertForTokenClassification.from_pretrained(bert_chinese_ner.model_path,config=self.bertconfig)

    @classmethod
    def getTokenizer(cls):
        return BertTokenizer.from_pretrained(bert_chinese_ner.model_path)

    def forward(self,input_ids,attention_mask,labels):
        '''
        input_ids:输入数据 [Batch_size,max_length]
        token_type_ids:该参数是当输入一个（sentence,sentence） tuple时标定word属于哪个句子，只能取1，0。这里可以不用
        attention_mask:Mask to avoid performing attention on padding token indices. Mask Value selected in [0,1] [Batch_size,max_length]
        labels:单词的tag,[Batch_size,max_length]
        '''
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)[0] # [] 0 是Loss 

    def test(self,input_ids,attention_mask):
        #score [batch_size,sentence_len,label_size] before Softmax
        return self.model(input_ids=input_ids,attention_mask=attention_mask)[0]


class BERT_Model(object):
    def __init__(self,vocab_size,out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        self.device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.vocab_size=vocab_size 
        self.emb_size=0
        self.hidden_size=0
        #因为使用Bert 预训练模型，因此该项没有作用, vocab_size defaults to 30522  hidden_size,embedding_size defaults yo 768
        self.out_size=out_size

        self.model=bert_chinese_ner(self.vocab_size,self.emb_size,self.hidden_size,self.out_size).to(self.device);
        self.tokenizer=bert_chinese_ner.getTokenizer()

        self.epoches=TrainingConfig.epoches
        self.print_step=TrainingConfig.print_step
        self.lr=TrainingConfig.lr
        self.batch_size=TrainingConfig.batch_size
        self.optimizer=optim.Adam(self.model.parameters(),lr=self.lr)

        self.step=0

        self._best_val_loss=10000
        self.best_model=None

    def train(self,train_word_lists,train_tag_lists,word2id,tag2id,dev_word_lists,dev_tag_lists):
        '''
        这里的train_word_lists,需要处理成[str,str,str]形式
        这里的word2id,没有使用，用的是BERT Tokenizer
        这里的Tag2id 是加入了"[CLS]","[SEP]"的，该tag再BERT中对句子的开头和结尾进行了标注
        '''
        #处理sentence
        train_sentence_list=[''.join(word_list) for word_list in train_word_lists]
        inputs=self.tokenizer(train_sentence_list,padding=True)
        train_sentence_list=inputs['input_ids']
        train_sentence_attention_mask=inputs['attention_mask']

        dev_sentence_list=[''.join(word_list) for word_list in dev_word_lists]
        inputs=self.tokenizer(dev_sentence_list,padding=True)
        dev_sentence_list=inputs['input_ids']
        dev_sentence_attention_mask=inputs['attention_mask']

        #处理Label
        max_len=len(train_sentence_list[0])
        train_tag_lists=tag_add_padding(train_tag_lists,max_len,tag2id)
        max_len=len(dev_sentence_list[0])
        dev_tag_lists=tag_add_padding(dev_tag_lists,max_len,tag2id)

        B=self.batch_size
        for e in range(1,self.epoches+1):
            self.step=0
            losses=0.
            
            for ind in range(0,len(train_sentence_list),B):
                batch_sents=train_sentence_list[ind:ind+B]
                batch_attention_mask=train_sentence_attention_mask[ind:ind+B]
                batch_tag=train_tag_lists[ind:ind+B]
                
                losses+=self.train_step(batch_sents,batch_attention_mask,batch_tag)

                if self.step % self.print_step==0:
                    total_step=(len(train_sentence_list))//self.batch_size +1
                    print("Epoch {},step/total_step: {}/{} Average Loss for one batch:{:.4}"\
                        .format(e,self.step,total_step,losses/self.print_step))
                    losses=0.
            val_loss=self.validate(dev_sentence_list,dev_sentence_attention_mask,\
                dev_tag_lists,word2id,tag2id)
            print("Epoch:{},Val_loss:{:4f}".format(e,val_loss))
            
    
    def train_step(self,batch_sents,batch_attention_mask,batch_tag):
        self.model.train()
        self.step+=1

        #这里使用from_numpy 因为pylint 一直误报错torch.tensor, 因此另辟蹊径
        batch_tensor_input_ids=torch.from_numpy(np.array(batch_sents)).long().to(self.device)
        batch_tensor_attention_mask=torch.from_numpy(np.array(batch_attention_mask)).float().to(self.device)
        batch_tensor_labels=torch.from_numpy(batch_tag).long().to(self.device)

        print(batch_tensor_input_ids.shape)
        print(batch_tensor_labels.shape)
        loss=self.model(input_ids=batch_tensor_input_ids,attention_mask=batch_tensor_attention_mask,\
            labels=batch_tensor_labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validate(self,sents_list,attention_mask,tag_lists,_,tag2id):
        self.model.eval()
        B=self.batch_size
        with torch.no_grad():
            val_losses=0.
            val_step=0
            for ind in range(0,len(sents_list),B):
                val_step+=1
                batch_sents=sents_list[ind:ind+B]
                batch_tag=tag_lists[ind:ind+B]
                batch_attention_mask=attention_mask[ind:ind+B]

                batch_tensor_input_ids=torch.from_numpy(np.array(batch_sents)).long().to(self.device)
                batch_tensor_attention_mask=torch.from_numpy(np.array(batch_attention_mask)).float().to(self.device)
                batch_tensor_labels=torch.from_numpy(batch_tag).long().to(self.device)
                
                loss=self.model(input_ids=batch_tensor_input_ids,attention_mask=batch_tensor_attention_mask,\
                    labels=batch_tensor_labels)

                val_losses+=loss.item()
            
            val_loss=val_losses/val_step

            if val_loss<self._best_val_loss:
                print("Upgrade Model and save Model")
                self.best_model=deepcopy(self.model) #deepcopy 深度复制，重新建立一个对象
                self._best_val_loss=val_loss
            return val_loss

    
    def test(self,test_word_lists,test_tag_lists,word2id,tag2id):
        
        test_sentence_list=[''.join(word_list) for word_list in test_word_lists]
        inputs=self.tokenizer(test_sentence_list,padding=True)
        test_sentence_list=inputs['input_ids']
        test_sentence_attention_mask=inputs['attention_mask']

        tensor_inputs_ids=torch.from_numpy(np.array(test_sentence_list)).long().to(self.device)
        tensor_attention_mask=torch.from_numpy(np.array(test_sentence_attention_mask)).float().to(self.device)
        
        max_len=len(test_sentence_list[0])
        tensor_labels=torch.from_numpy(tag_add_padding(test_tag_lists,max_len,tag2id)).long().to(self.device)

        mask=(tensor_labels!=tag2id.get('[CLS]')) & (tensor_labels!=tag2id.get('[SEP]')) \
            & (tensor_labels!=tag2id.get('[PAD]'))

        self.best_model.eval()

        with torch.no_grad():
            outputs=self.best_model.test(tensor_inputs_ids,tensor_attention_mask)
            #outputs [Batch_size, sentence_len,label_num]
            out=torch.argmax(outputs,dim=2) #[Batch_size,sentence_len]
            pred_tagid=out.masked_select(mask)

        #将tagid 解码成tag
        id2tag=dict((value,key) for key,value in tag2id.items())
        pred_tag=[id2tag.get(i.item()) for i in pred_tagid]
        
        return pred_tag #list 


            

