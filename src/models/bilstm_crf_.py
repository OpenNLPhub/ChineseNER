

import os
import sys
sys.path.append(os.path.join(os.getcwd(),'src'))

import torch
from .crf import CRF
from .bilstm import BiLSTM
from torch import nn
from config import LSTMConfig,BiLSTM_CRF_TrainingConfig
from utils import tensorized,sort_by_lengths,cal_loss,cal_lstm_crf_loss
import torch.optim as optim
from copy import deepcopy
class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size,emb_size,hidden_size,out_size):
        super(BiLSTM_CRF,self).__init__()
        self.bilstm=BiLSTM(vocab_size,emb_size,hidden_size,out_size)
        self.crf=CRF(out_size)


    def forward(self,sents_tensor,lengths):
        # [B,L,out_size]
        emission=self.bilstm(sents_tensor,lengths)
        return emission
    
    def loss_cal(self,bilstm_outputs,length,labels):
        loss=self.crf.loss_cal(bilstm_outputs,length,labels)

    def test(self,sents_tensor,length):
        emission=self.bilstm(sents_tensor,length)
        batch_best_path=self.crf.get_batch_best_path(emission,length)
        return batch_best_path
    
class BiLSTM_CRF_Model(object):
    def __init__(self,vocab_size,out_size):
        self.device="cuda:1" if torch.cuda.is_available() else "cpu"
        self.embedding_size=LSTMConfig.embedding_size
        self.hidden_size=LSTMConfig.hidden_size
        self.model=BiLSTM_CRF(vocab_size,self.embedding_size,self.hidden_size,out_size).to(self.device)

        self.epoches=BiLSTM_CRF_TrainingConfig.epoches
        self.print_step=BiLSTM_CRF_TrainingConfig.print_step
        self.lr=BiLSTM_CRF_TrainingConfig.lr
        self.batch_size=BiLSTM_CRF_TrainingConfig.batch_size

        self.optimizer=optim.Adam(self.model.parameters(),lr=self.lr)

        self.step=0
        #最佳损失函数,初始化一个极大值
        self._best_val_loss=1e18
        self.best_model=None
    
    def train(self,train_word_lists,train_tag_lists,word2id,tag2id,dev_word_lists,dev_tag_lists):
        #按句子长短进行排序
        #训练集,验证集中句子顺序无需还原
        train_word_lists,train_tag_lists, _ =sort_by_lengths(train_word_lists,train_tag_lists)
        dev_word_lists,dev_tag_lists,_=sort_by_lengths(dev_word_lists,dev_tag_lists)

        B=self.batch_size

        for e in range(1,self.epoches+1):
            #迭代轮次
            self.step=0
            losses=0.
            for ind in range(0,len(train_tag_lists),B):
                #每次训练B个句子
                batch_sents=train_word_lists[ind:ind+B]
                batch_tag=train_tag_lists[ind:ind+B]

                losses+=self.train_step(batch_sents,batch_tag,word2id,tag2id)

                if self.step % BiLSTM_CRF_TrainingConfig.print_step == 0:
                    total_step=(len(train_word_lists)//self.batch_size +1)
                    print("Epoch {}, step/total_step: {}/{} Average Loss for one batch:{:.4f}".format(e,self.step,total_step,losses/self.print_step))
                    losses=0.

            # val_loss=self.validate(dev_word_lists,dev_tag_lists,word2id,tag2id)
            # print("Epoch {}, Val Loss:{:.4f}".format(e,val_loss))
    

    def train_step(self,word_lists,tag_lists,word2id,tag2id):
        self.model.train()
        self.step+=1

        #lengths 相同
        tensorized_sents,lengths=tensorized(word_lists,word2id)
        tensorized_tags,lengths=tensorized(tag_lists,tag2id)

        tensorized_sents=tensorized_sents.to(self.device)
        tensorized_tags=tensorized_tags.to(self.device)
        scores=self.model(tensorized_sents,lengths)

        #计算损失
        self.optimizer.zero_grad()
        loss=self.model.loss_cal(scores,lengths,tensorized_tags)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    #验证
    def validate(self,dev_word_lists,dev_tag_lists,word2id,tag2id):
        self.model.eval()
        #不追踪梯度，节省内存
        with torch.no_grad():
            val_losses=0.
            val_step=0
            for ind in range(0,len(dev_word_lists),self.batch_size):
                val_step+=1
                batch_sents=dev_word_lists[ind:ind+self.batch_size]
                batch_tag=dev_tag_lists[ind:ind+self.batch_size]
                tensorized_sent,lengths=tensorized(batch_sents,word2id)
                tensorized_tag,lengths=tensorized(batch_tag,tag2id)

                tensorized_sent=tensorized_sent.to(self.device)
                tensorized_tag=tensorized_tag.to(self.device)


                scores=self.model(tensorized_sent,lengths)
                loss=self.model.loss_cal(scores,lengths,tensorized_tag)
                val_losses+=loss.item()
            val_loss=val_losses/val_step/self.batch_size  #每个Batch随机损失值

            if val_loss<self._best_val_loss:
                print("Upgrade Model and save Model")

                self.best_model=deepcopy(self.model) #deepcopy 深度复制，重新建立一个对象
                self._best_val_loss=val_loss

            return val_loss
    
    def test(self,test_word_lists,test_tag_lists,word2id,tag2id):
        test_word_lists,test_tag_lists,indices= sort_by_lengths(test_word_lists,test_tag_lists)
        tensorized_sent,lengths=tensorized(test_word_lists,word2id)
        tag_lists=[ test_tag_list[:lengths[i]] for i,test_tag_list in enumerate(test_tag_lists)]
        self.best_model.eval()
        pred_tagid_lists=[]
        with torch.no_grad():
            B=self.batch_size
            for ind in range(0,len(test_word_lists),B):
                tensorized_batch_sent=tensorized_sent.to(self.device)
                batch_tagids=self.best_model.test(tensorized_batch_sent,lengths,tag2id) #[B,L]
                pred_tagid_lists+=batch_tagids
        
        id2tag=dict((id,tag) for tag,id in tag2id.items())
        pred_tag_lists=[] #[B,L]
        for i,ids in enumerate(pred_tagid_lists):
            tag_list=[]
            for j in range(lengths[i]):
                tag_list.append(id2tag.get(ids[j]))
            pred_tag_lists.append(tag_list)

        return pred_tag_lists,tag_lists