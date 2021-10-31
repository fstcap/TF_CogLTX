import logging
import pickle
import random
from tqdm import tqdm  # 引入进度条模块
import tensorflow as tf
from utils import CAPACITY, BLOCK_SIZE, ESTIMATIONS_FILE_PATH
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
        n0, n1 = [int(s) for s in num_samples.split(',')][:2]  # 1, 1
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)  # 模型最大长度限制下分得组数
        logging.info('building buffers for introspection...')
        for qbuf, dbuf in tqdm(self.dataset):
            # 1. continous
            lb = max_blk_num - len(qbuf)  # 组数减去问题的block的长度，剩下就是段可以加入的block的长度
            st = random.randint(0, max(0, len(dbuf) - lb * n0))  # 随机一个index
            for i in range(n0):
                buf = Buffer()
                buf.blocks = qbuf.blocks + dbuf.blocks[st + i * lb:st + (i + 1) * lb]  # 获取一个随机的问题段落block的组合
                ret.append(buf)
            # 2. pos + neg
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            for i in range(n1):
                selected_pblks = random.sample(pbuf.blocks, min(lb, len(pbuf)))
                selected_nblks = random.sample(nbuf.blocks, min(lb - len(selected_pblks), len(nbuf)))
                buf = Buffer()
                buf.blocks = qbuf.blocks + selected_pblks + selected_nblks  # 随机一个含有最多正样本其余为负样本的组合
                ret.append(buf.sort_())
        return SimpleListDataset(ret)

    def collect_estimations_from_dir(self):
        """读取分类模型预测每个block分数
        赋值给Buffer类中的属性d列表每个元素的estimation属性
        :return:
        """
        with open(ESTIMATIONS_FILE_PATH, 'r') as fin:
            for line in fin:
                ls = line.split()
                pos, estimation = int(ls[0]), float(ls[1])
                self.d[pos].estimation = estimation

    def build_promising_buffer(self, num_samples):
        """筛选出block.relevance>1的样本与block.relevance=0且按照estimation降序的样本组合
        和block.relevance>1的样本与block.relevance=0随机组合的样本
        :param num_samples:
        :return:
        """
        n2, n3 = [int(x) for x in num_samples.split(',')][2:]
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        logging.info('为reasoning创建数据...')
        for qbuf, dbuf in tqdm(self.dataset):
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            if len(pbuf) >= max_blk_num - len(qbuf):
                pbuf = pbuf.random_sample(max_blk_num - len(qbuf) - 1)
            lb = max_blk_num - len(qbuf) - len(pbuf)
            estimations = tf.constant([blk.estimation for blk in nbuf])
            keeped_indices = tf.argsort(estimations, direction='DESCENDING')[:n2 * lb]
            selected_nblks = [blk for i, blk in enumerate(nbuf) if i in keeped_indices]
            while 0 < len(selected_nblks) < n2 * lb:
                selected_nblks = selected_nblks * (n2 * lb // len(selected_nblks) + 1)  # 如果选出负样本不够n2 * lb的长度，复制样本加长
            for i in range(n2):
                buf = Buffer()
                buf.blocks = qbuf.blocks + pbuf.blocks + selected_nblks[i * lb: (i + 1) * lb]
                ret.append(buf.sort_())
            for i in range(n3):
                buf = Buffer()
                buf.blocks = qbuf.blocks + pbuf.blocks + random.sample(nbuf.blocks, min(len(nbuf), lb))
                ret.append(buf.sort_())
        return SimpleListDataset(ret)
