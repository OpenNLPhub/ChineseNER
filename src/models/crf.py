import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
class CRF(nn.Module):
    '''条件随机场
       因为在组合模型中一次又一次 重写CRF代码，于是用重写这个类，将模块抽离出来
       tag_nums: label nums，对应转移矩阵的大小
       is_padding 对于没有Start End 或类似开始结束标签的，需要打上标签,
                  True 输入的tag_num里面已经含有了标签
        start_tag,end_tag 若is_padding 为Ture,start_tag和end_tag为在tag_nums中的索引
    '''
    def __init__(self,tag_nums,is_padding=False,start_tag=None,end_tag=None):
        super(CRF,self).__init__()
        self.tag_nums=tag_nums
        if is_padding:
            self.start_tag=start_tag
            self.end_tag=end_tag
            self.trans_size=self.tag_nums
        else:
            self.start_tag=self.tag_nums
            self.end_tag=self.tag_nums+1
            self.trans_size=self.tag_nums+2
        
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        self.transitions=nn.Parameter(torch.from_numpy(np.random.randn(self.trans_size,self.trans_size))).to(self.device)
        #self.transitions[i][j] 表示从隐藏状态i转换到隐藏状态j的概率
        self.transitions[:,self.start_tag]=-1000 #从其他状态到START状态，为不可能事件
        self.transitions[self.end_tag,:]=-1000 #从END状态 到其他状态，为不可能事件
    
    '''损失函数计算
       features [batch_size,length,num_tag]
       length [batch_size,len]
       target_function= real_path_score / all_possible_path_score
                      = exp(S_real_path_score) / sum(exp(certain_path_score)
       对于Target_function 希望real_path_score 得分更高
       因此损失函数增加一个 ‘-’
       loss_function= -log(target_function)= -S_real_path_score + log(sum(exp(certain_path_score)))

       S_real_path_score 通过函数 _get_real_path_score 获得
       log sum(exp(S1)+exp(S2)+exp(S3)+...+exp(Sn))通过 _get_all_possible_path_score 获得 
    '''
    def loss_cal(self,features,length,tags):
        batch_size,_,num_tag=features.shape
        loss=torch.zeros(1,requires_grad=True).squeeze(0)

        #feature [padding_length,num_tag]
        #tag [padding_length]
        for ix,(feature,tag) in enumerate(zip(features,tags)):
            word_nums=length[ix] #不加padding
            feature=feature[:word_nums]
            tag=tag[:word_nums]
            
            assert len(feature)==len(tag)

            real_score=self._get_real_path_score(feature,tag)
            total_score=self._get_all_possible_path_score(feature)

            cost=total_score-real_score

            loss+=cost
    
        return loss/batch_size

    '''计算真实路径得分
        feature [length_sent,num_tag]
        tagid [length_sent]
    '''
    def _get_real_path_score(self,feature,tagid):
        # [start_tag_id, tag1id,tag2id,...]
        tag_padding_start=torch.cat([torch.from_numpy(np.array([self.start_tag])).to(self.device),tagid])
        # [tag1id,tag2id,...,tagnid,end_tag_id]
        tag_padding_end=torch.cat([tagid,torch.from_numpy(np.array([self.end_tag])).to(self.device)])

        trans_score=self.transitions[tag_padding_start,tag_padding_end]
        trans_score=torch.sum(trans_score)

        order=torch.LongTensor(range(feature.size(0))).to(self.device)

        emission_score=torch.sum(feature[order,tagid])

        return emission_score+trans_score
    
    '''计算所有路径得分
        前向算法 forward algorithm
        feature [length_sent,num_tag]
        
    '''
    def _get_all_possible_path_score(self,feature):
        words_num=feature.size(0)
        forward=torch.zeros(self.tag_nums).to(self.device)
        
        #START 初始化 START->tag1
        emission_start=forward
        emission_end=feature[0]
        transition_score=self.transitions[self.start_tag,:self.tag_nums]
        score=emission_start+transition_score+emission_end
        forward=self._log_sum(score.unsqueeze(0),dim=0)
        # [tag_nums]

        # tag1->tag2->tag3->...->tagN
        for ix in range(1,words_num):
            emission_start=forward.expand(self.tag_nums,self.tag_nums).T
            emission_end=feature[ix].expand(self.tag_nums,self.tag_nums)
            transition_score=self.transitions[:self.tag_nums,:self.tag_nums]
            score=emission_start+emission_end+transition_score
            # [tag_nums,tag_nums] 
            forward=self._log_sum(score,dim=0)
        
        #tagN->END forward size=(tag_nums)
        forward=forward+self.transitions[:self.tag_nums:,self.end_tag]

        total_score=self._log_sum(forward,dim=0)
        
        return total_score
    

    '''
    [tag_nums,tag_nums]的矩阵，对行求和 [tag_nums],第i个元素代表盈仓状态i对应的概率和
    这里 考虑到 exp 在前向算法中指数累计到一定程度后，会超过计算机浮点数最大值，上溢出（虽然每一个概率都小于1）
    为了避免这样的情况，用一个最大值去提取指数公因子
    SUM=log(exp(s1)+exp(s2)+...exp(sN))
       =log{exp(clip)*(exp(s1-clip)+exp(s2-clip)+...+exp(sN-clip))}
       =log(exp(clip))+log(exp(s1-clip)+exp(s2-clip)+...+exp(sN-clip))
       =clip+log(exp(s1-clip)+exp(s2-clip)+...+exp(sN-clip))
    '''

    @classmethod
    def _log_sum(cls,s,dim=0):
        clip=torch.max(s)
        log_sum_value=clip+torch.log(torch.sum(torch.exp(s-clip),dim=dim))
        return s


    ''' 维特比解码 '''
    '''
    维特比解码
    features [ word_nums , tag_nums]
    '''
    def _viberbi(self,features):
        words_num=features.size(0)
        preview=torch.zeros(self.tag_nums).to(self.device)
        backpointer=torch.zeros(words_num,self.tag_nums).long().to(self.device)
        
        #START 初始化 START->tag1
        emission_start=preview
        trans_score=self.transitions[self.start_tag,:self.tag_nums]
        emission_end=features[0]
        score=emission_start+trans_score+emission_end
        preview=self._log_sum(score.unsqueeze(0),dim=0)

        # tag1->tag2->...->tagN
        for ix in range(1,words_num):
            emission_start=preview.expand(self.tag_nums,self.tag_nums).T
            trans_score=self.transition[:self.tag_nums,:self.tag_nums]
            emission_end=features[ix].expand(self.tag_nums,self.tag_nums)
            score=emission_start+trans_score+emission_end
            #score [tag_nums,tag_nums] 
            #其中 score[i][j]代表从 ix-1时刻隐藏状态为 i 到 ix时刻 隐藏状态为j

            backpointer[ix]=torch.argmax(score,dim=0)

            #[tag_nums,tag_nums]
            preview=self._log_sum(score,dim=0)
        

        #开始回溯
        best_path=[]
        backpoint=torch.argmax(preview).item()
        best_path.append(backpoint)
        for ix in range(words_num-1,0,-1):
            backpoint=backpointer[ix][backpoint]
            best_path.append(backpoint)
        return reversed(best_path)

    '''
    inputs [batch_size,padding_length,tag_nums]
    length [batch_size] 记录每个句子的长度
    '''
    def get_batch_best_path(self,inputs,lengths):
        assert len(inputs)==len(lengths)
        batch_best_path=[]
        for ix,features in enumerate(inputs):
            length=lengths[ix]
            features=features[:length]
            best_path=self._viberbi(features)
            batch_best_path.append(best_path)
        return batch_best_path


        
        
            









