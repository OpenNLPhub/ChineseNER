'''
 * @author Waileinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-08-05 15:53:25
 * @modify date 2020-08-05 15:53:25
'''


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

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids


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
        self.device=torch.device("cuda" if torch.cuda.is_availabel() else "cpu")

        self.embedding_size=0
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
        loss=self.cal_loss_func(scores,tag_lists,tag2id).to(self.device)
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
                batch_sents=dev_word_lists[ind,ind+self.batch_size]
                batch_tag=dev_tag_lists[ind,ind+self.batch_size]
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
        for i,ids in batch_tagids:
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





'''
Hidden Markov Model
隐马尔可夫模型
'''

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
        

class HMM_standard(object):
    def __init__(self, N, M):
        """Args:
            N: 状态数，这里对应存在的标注的种类
            M: 观测数，这里对应有多少不同的字
        """
        self.N = N
        self.M = M

        # 状态转移概率矩阵 A[i][j]表示从i状态转移到j状态的概率
        self.A = torch.zeros(N, N)
        # 观测概率矩阵, B[i][j]表示i状态下生成j观测的概率
        self.B = torch.zeros(N, M)
        # 初始状态概率  Pi[i]表示初始时刻为状态i的概率
        self.Pi = torch.zeros(N)

    def train(self, word_lists, tag_lists, word2id, tag2id):
        """HMM的训练，即根据训练语料对模型参数进行估计,
           因为我们有观测序列以及其对应的状态序列，所以我们
           可以使用极大似然估计的方法来估计隐马尔可夫模型的参数
        参数:
            word_lists: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
            tag_lists: 列表，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
            word2id: 将字映射为ID
            tag2id: 字典，将标注映射为ID
        """

        assert len(tag_lists) == len(word_lists)

        # 估计转移概率矩阵
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i+1]]
                self.A[current_tagid][next_tagid] += 1
        # 问题：如果某元素没有出现过，该位置为0，这在后续的计算中是不允许的
        # 解决方法：我们将等于0的概率加上很小的数
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        # 估计观测概率矩阵
        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        # 估计初始状态概率
        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def test(self, word_lists, word2id, tag2id):
        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_list = self.decoding(word_list, word2id, tag2id)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists

    def decoding(self, word_list, word2id, tag2id):
        """
        使用维特比算法对给定观测序列求状态序列， 这里就是对字组成的序列,求其对应的标注。
        维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径）
        这时一条路径对应着一个状态序列
        """
        # 问题:整条链很长的情况下，十分多的小概率相乘，最后可能造成下溢
        # 解决办法：采用对数概率，这样源空间中的很小概率，就被映射到对数空间的大的负数
        #  同时相乘操作也变成简单的相加操作
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)

        # 初始化 维比特矩阵viterbi 它的维度为[状态数, 序列长度]
        # 其中viterbi[i, j]表示标注序列的第j个标注为i的所有单个序列(i_1, i_2, ..i_j)出现的概率最大值
        seq_len = len(word_list)
        viterbi = torch.zeros(self.N, seq_len)
        # backpointer是跟viterbi一样大小的矩阵
        # backpointer[i, j]存储的是 标注序列的第j个标注为i时，第j-1个标注的id
        # 等解码的时候，我们用backpointer进行回溯，以求出最优路径
        backpointer = torch.zeros(self.N, seq_len).long()

        # self.Pi[i] 表示第一个字的标记为i的概率
        # Bt[word_id]表示字为word_id的时候，对应各个标记的概率
        # self.A.t()[tag_id]表示各个状态转移到tag_id对应的概率

        # 所以第一步为
        start_wordid = word2id.get(word_list[0], None)
        Bt = B.t()
        if start_wordid is None:
            # 如果字不再字典里，则假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.N) / self.N)
        else:
            bt = Bt[start_wordid]
        viterbi[:, 0] = Pi + bt
        backpointer[:, 0] = -1

        # 递推公式：
        # viterbi[tag_id, step] = max(viterbi[:, step-1]* self.A.t()[tag_id] * Bt[word])
        # 其中word是step时刻对应的字
        # 由上述递推公式求后续各步
        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step], None)
            # 处理字不在字典中的情况
            # bt是在t时刻字为wordid时，状态的概率分布
            if wordid is None:
                # 如果字不再字典里，则假设状态的概率分布是均匀的
                bt = torch.log(torch.ones(self.N) / self.N)
            else:
                bt = Bt[wordid]  # 否则从观测概率矩阵中取bt
            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(
                    viterbi[:, step-1] + A[:, tag_id],
                    dim=0
                )
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                backpointer[tag_id, step] = max_id

        # 终止， t=seq_len 即 viterbi[:, seq_len]中的最大概率，就是最优路径的概率
        best_path_prob, best_path_pointer = torch.max(
            viterbi[:, seq_len-1], dim=0
        )

        # 回溯，求最优路径
        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len-1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        # 将tag_id组成的序列转化为tag
        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list  
    


    








if __name__=='__main__':
    print(LSTMConfig.embedding_size)