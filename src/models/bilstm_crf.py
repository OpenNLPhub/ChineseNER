import sys
import os
sys.path.append(os.path.join(os.getcwd(),'src'))


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from config import LSTMConfig,TrainingConfig
from itertools import zip_longest
from copy import deepcopy
from utils import tensorized,sort_by_lengths,cal_loss,cal_lstm_crf_loss
from models.bilstm import BiLSTM

class BiLSTM_CRF(nn.Module):
    """
    Copy from PyTorch Tutorial
    """
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)
        # self.transition.data.zero_()

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size]
        emission = self.bilstm(sents_tensor, lengths)

        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids


class BiLSTM_CRF_Model():
    def __init__(self,vocab_size,out_size,crf=True):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_size=LSTMConfig.embedding_size
        self.hidden_size=LSTMConfig.hidden_size

        self.crf=crf

        #无条件随机场
        if not crf:
            self.model=BiLSTM(vocab_size,self.embedding_size,self.hidden_size,out_size).to(self.device)
            self.cal_loss_func=cal_loss
        else:
            self.model=BiLSTM_CRF(vocab_size,self.embedding_size,self.hidden_size,out_size).to(self.device)
            self.cal_loss_func=cal_lstm_crf_loss

        self.epoches=TrainingConfig.epoches
        self.print_step=TrainingConfig.print_step
        self.lr=TrainingConfig.lr
        self.batch_size=TrainingConfig.batch_size

        self.optimizer=optim.Adam(self.model.parameters(),lr=self.lr)

        self.step=0
        #最佳损失函数,初始化一个极大值
        self._best_val_loss=1e18
        self.best_model=None


    #Train 训练集
    #Dev 验证集
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

                if self.step % TrainingConfig.print_step == 0:
                    total_step=(len(train_word_lists)//self.batch_size +1)
                    print("Epoch {}, step/total_step: {}/{} Average Loss for one batch:{:.4f}".format(e,self.step,total_step,losses/self.print_step))
                    losses=0.

            val_loss=self.validate(dev_word_lists,dev_tag_lists,word2id,tag2id)
            print("Epoch {}, Val Loss:{:.4f}".format(e,val_loss))


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
        loss=self.cal_loss_func(scores,tensorized_tags,tag2id).to(self.device)
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
                loss=self.cal_loss_func(scores,tensorized_tag,tag2id).to(self.device)

                val_losses+=loss.item()
            val_loss=val_losses/val_step  #每个Batch随机损失值

            if val_loss<self._best_val_loss:
                print("Upgrade Model and save Model")

                self.best_model=deepcopy(self.model) #deepcopy 深度复制，重新建立一个对象
                self._best_val_loss=val_loss

            return val_loss


    #测试
    def test(self,test_word_lists,test_tag_lists,word2id,tag2id):

        #要还原句子顺序
        test_word_lists,test_tag_lists,indices= sort_by_lengths(test_word_lists,test_tag_lists)

        tensorized_sent,lengths=tensorized(test_word_lists,word2id)
        tensorized_tag,lengths=tensorized(test_word_lists,tag2id)

        tensorized_sent=tensorized_sent.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids=self.best_model.test(tensorized_sent,lengths,tag2id) #[B,L]
        id2tag=dict((id,tag) for tag,id in tag2id.items())
        pred_tag_lists=[] #[B,L]
        for i,ids in enumerate(batch_tagids):
            tag_list=[] #(L,)
            if self.crf:
                for j in range(lengths[i]-1):
                    tag_list.append(id2tag[ids[j].item()]) #item() 取 tensor中的值，容易忘记
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])

            pred_tag_lists.append(tag_list)

        #indices= [1,2,0] 表示 原先索引为1的 新的索引是0 [(0,1) (1,2),(2,0)] 排序后 [(2,0),(0,1),(1,2)]
        ind_maps=sorted(list(enumerate(indices)),key=lambda e:e[1])
        indices,_=list(zip(*ind_maps))
        pred_tag_lists=[pred_tag_lists[i] for i in indices]
        tag_lists=[test_tag_lists[i] for i in indices]

        return pred_tag_lists,tag_lists

    #评估句子
    def test_no_tag(self,word_lists,word2id,tag2id):
        indices=sorted(range(len(word_lists)),key=lambda k:len(word_lists[k]),reversed=True)
        word_lists=[word_lists[i] for i in indices]

        tensorized_sent,lengths=tensorized(word_lists,word2id)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids=self.best_model.test(tensorized_sent,lengths,tag2id) #[B,L]
        id2tag=dict((id,tag) for tag,id in tag2id.items())

        #将id转化为tag
        pred_tag_lists=[]
        for i, ids in batch_tagids:
            tag_list=[]
            if self.crf:
                for j in range(lengths[i]-1):
                    tag_list.append(id2tag[ids[j].item()])
            else:
                tag_list.append(id2tag[ids[j].item()])

        #将句子顺序还原

        ind_maps=sorted(list(enumerate(indices)),kkey=lambda e:e[1])
        indices,_=list(zip(*ind_maps))

        pred_tag_lists=[pred_tag_lists[i] for i in indices]
        word_lists=[word_lists[i] for i in indices]

        return pred_tag_lists,word_lists

    #打印混淆矩阵
    #多分类问题
    def print_confusion_matrix(self,pred_tag_lists,tag_lists):
        pass

