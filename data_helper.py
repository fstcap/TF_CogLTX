import logging
import pickle
import random
from tqdm import tqdm # 引入进度条模块
from utils import CAPACITY, BLOCK_SIZE
from buffer import Buffer

class SimpleListDataset:
    """从硬盘或直接列表载入
    """
    def __init__(self, source):
        if isinstance(source, str):
            with open(source, 'rb') as fin:
                logging.info('Loading dataset...')
                self.dataset = pickle.load(fin)
        elif isinstance(source, list):
            self.dataset = source
        if not isinstance(self.dataset, list):
            raise ValueError('The source of SimpleListDataset is not a list.')
    def __getitem__(self, index):
        """用于类遍历
        :param index:
        :return:
        """
        return self.dataset[index]
    def __len__(self):
        """获取类长度
        :return:
        """
        return len(self.dataset)

class BlkPosInterface:
    def __init__(self, dataset):
        assert isinstance(dataset, SimpleListDataset)
        self.d = {}
        self.dataset = dataset
        for bufs in dataset:
            for buf in bufs:
                for blk in buf:
                    assert blk.pos not in self.d
                    self.d[blk.pos] = blk

    def build_random_buffer(self, num_samples):
        """每个问题中，获取一个随机样本和一个由最多正样本blocks其余为负样本blocks的组合
        :param num_samples:
        :return:
        """
        n0, n1 = [int(s) for s in num_samples.split(',')][:2] # 1, 1
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1) # 模型最大长度限制下分得组数
        logging.info('building buffers for introspection...')
        for qbuf, dbuf in tqdm(self.dataset):
            # 1. continous
            lb = max_blk_num - len(qbuf) # 组数减去问题的block的长度，剩下就是段可以加入的block的长度
            st = random.randint(0, max(0, len(dbuf) - lb * n0)) # 随机一个index
            for i in range(n0):
                buf = Buffer()
                buf.blocks = qbuf.blocks + dbuf.blocks[st + i * lb:st + (i+1) * lb] # 获取一个随机的问题段落block的组合
                ret.append(buf)
            # 2. pos + neg
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            for i in range(n1):
                selected_pblks = random.sample(pbuf.blocks, min(lb, len(pbuf)))
                selected_nblks = random.sample(nbuf.blocks, min(lb - len(selected_pblks), len(nbuf)))
                buf = Buffer()
                buf.blocks = qbuf.blocks + selected_pblks + selected_nblks # 随机一个含有最多正样本其余为负样本的组合
                ret.append(buf.sort_())
        return SimpleListDataset(ret)