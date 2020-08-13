import sys
import os
sys.path.append(os.path.join(os.getcwd(),'src'))


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import progressbar
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from config import LSTMConfig,TrainingConfig
from itertools import zip_longest
from copy import deepcopy
from utils import tensorized,sort_by_lengths,cal_loss,cal_lstm_crf_loss


class HMM(object):
    def __init__(self,N,M):
        '''
        N 隐藏状态数
        M 观测状态数
        '''
        self.N=N
        self.M=M

        #torch tensor 默认类型 float32
        #状态转移矩阵
        self.A=torch.zeros(N,N)
        #观测产生矩阵
        self.B=torch.zeros(N,M)
        #初始状态概率分布
        self.Pi=torch.zeros(N)

        #在进行解码的时候涉及很多小数概率相乘，我们对其取对数
        self.logA=torch.zeros(N,N)
        self.logB=torch.zeros(N,M)
        self.logPi=torch.zeros(N)

    def train(self,word_lists,tag_lists,word2id,tag2id):
        '''
        word_lists = 句子列表 [ [ '北','京','欢','迎','你'], [ ],[ ]]
        tag_lists [ ['B-LOC','I-LOC','O','O','O'],[ ],[ ],[ ] ]
        word2id 映射到index
        tag2id 映射到index
        '''
        assert len(word_lists)==len(tag_lists)
        #统计转移矩阵 和 初始概率分布矩阵
        for tag_list in tag_lists:
            l=len(tag_list)-1
            for j in range(l):
                next_tag_id=tag2id[tag_list[j+1]]
                tag_id=tag2id[tag_list[j]]
                self.A[tag_id][next_tag_id]+=1
                self.Pi[tag_id]+=1
                if j==l-1: self.Pi[next_tag_id]+=1 
        Asum=torch.sum(self.A,1).unsqueeze(1)
        self.A=self.A/Asum
        pisum=torch.sum(self.Pi)
        self.Pi=self.Pi/pisum
        
        #统计生成矩阵
        for i in range(len(tag_lists)):
            tag_list=tag_lists[i]
            word_list=word_lists[i]
            for j in range(len(tag_list)):
                tag_id=tag2id[tag_list[j]]
                word_id=word2id[word_list[j]]
                self.B[tag_id][word_id]+=1
        Bsum=torch.sum(self.B,1).unsqueeze(1)
        self.B=self.B/Bsum

        self.logA=torch.log(self.A)
        self.logB=torch.log(self.B)
        self.logPi=torch.log(self.Pi)

    def test(self,test_word_lists,_,word2id,tag2id):
        pred_tag_lists=[]
        # for test_word_list in test_word_lists:
        #     pred_tag_list=self.decoding(test_word_list,word2id,tag2id)
        #     pred_tag_lists.append(pred_tag_list)
        for i in progressbar.progressbar(range(len(test_word_lists))):
            test_word_list=test_word_lists[i]
            pred_tag_list=self.decoding(test_word_list,word2id,tag2id)
            pred_tag_lists.append(pred_tag_list)

        return pred_tag_lists


    def decoding(self,word_list,word2id,tag2id):
        '''
        使用维特比算法进行状态序列求解
        '''
        length=len(word_list)
        
        #定义delta[t][n]记录 t 时刻 隐藏状态为n的 概率最大值
        delta=torch.zeros(length,self.N)
        #定义Psi[t][n] 当t时刻，隐藏状态为n，概率最大路径上 t-1 时的 隐藏状态
        psi=torch.zeros(length,self.N).long()

        #进行转置，便于并行计算
        Bt=self.logB.t()
        At=self.logA.t()

        #初始化 递推状态
        first_word_id=word2id.get(word_list[0],None)
        if first_word_id==None:
            #word UNK 字典里不存在,认为隐藏状态的分布是平均分布
            bt=torch.log(torch.ones(self.N)/self.N)
        else:
            bt=Bt[first_word_id]

        delta[0]=self.logPi+bt
        psi[0]=torch.zeros(self.N).long()

        #开始递推
        #递推公式 
        for t in range(1,length):
            word_id=word2id.get(word_list[t],None)
            if word_id==None:
                bt=torch.log(torch.ones(self.N)/self.N)
            else:
                bt=Bt[word_id]
            
            for i in range(self.N):
                at=At[i] # 1,2,...,N 转到 状态i 的转移概率 向量
                dd=delta[t-1] # 1,2,...,N 最大概率
                tmp=at+dd

            dd=delta[t-1] 
            tmp=At+dd # max[ delta[t-1] * a]
            delta[t],psi[t]=torch.max(tmp,dim=1) #计算最大概率
            delta[t]+=bt
        
        best_path=[]
        #使用回溯法，找到最佳隐藏序列

        #最后一个单词对应的隐藏状态
        i_=torch.argmax(delta[length-1]).item()
        best_path.append(i_)
        for t in range(length-1,0,-1):
            i_=psi[t][i_].item()
            best_path.append(i_)
        
        id2tag=dict((id_,tag) for tag,id_ in tag2id.items())
        best_path=[ id2tag[id_] for id_ in reversed(best_path)]

        return best_path
     