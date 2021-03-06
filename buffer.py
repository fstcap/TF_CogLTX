import random
import numpy as np
from transformers import RobertaTokenizer
from utils import CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME  # 默认模型roberta-base


class Block:
    tokenizer = RobertaTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    def __init__(self, ids, pos, blk_type=1, **kwargs):
        self.ids = ids
        self.pos = pos
        self.blk_type = blk_type  # 0 sentence A, 1 sentence B
        self.relevance = 0
        self.estimation = 0
        self.__dict__.update(kwargs)

    def __lt__(self, rhs):
        """
        逻辑运算小于，比较两个类的pos大小按快的原有顺序排列
        :param rhs:
        :return:
        """
        return self.blk_type < rhs.blk_type or (self.blk_type == rhs.blk_type and self.pos < rhs.pos)

    def __ne__(self, rhs):
        """
        类的逻辑运算不相等
        :param rhs:
        :return:
        """
        return self.pos != rhs.pos or self.blk_type != rhs.blk_type

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        """
        显示块代表的字符串
        :return:
        """
        return Block.tokenizer.convert_tokens_to_string(Block.tokenizer.convert_ids_to_tokens(self.ids))


class Buffer:
    @staticmethod
    def split_document_into_blocks(d, tokenizer, cnt=0, hard=True, properties=None):
        """分割段成block
        :param d:
        :param tokenizer:
        :param cnt:
        :param hard:
        :param properties:
        :return:
        """

        ret = Buffer()
        updiv = lambda a, b: (a - 1) // b + 1
        if hard:
            for sid, tsen in enumerate(d):
                psen = properties[sid] if properties is not None else []
                num = updiv(len(tsen), BLOCK_SIZE)  # 计算以63长度每快，分组加一
                bsize = updiv(len(tsen), num)  # 最终有num组，每块长度是bsize
                for i in range(num):
                    st, en = i * bsize, min((i + 1) * bsize, len(tsen))  # 每块的起始位置和结束位置
                    cnt += 1
                    tmp = tsen[st: en] + [tokenizer.sep_token]  # 每块的token列表
                    # inject properties into blks
                    tmp_kwargs = {}
                    for p in psen:
                        if len(p) == 2:
                            tmp_kwargs[p[0]] = p[1]
                        elif len(p) == 3:
                            if st <= p[1] < en:
                                tmp_kwargs[p[0]] = (p[1] - st, p[2])
                        else:
                            raise ValueError('Invalid property {}'.format(p))
                    block = Block(tokenizer.convert_tokens_to_ids(tmp), cnt, **tmp_kwargs)
                    ret.insert(block)

        return ret, cnt

    def __init__(self):
        self.blocks = []

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, key):
        return self.blocks[key]

    def sort_(self):
        self.blocks.sort()
        return self

    def block_ends(self):
        """计算每个block结尾的index+1
        :return:
        """
        t, ret = 0, []
        for b in self.blocks:
            t += len(b)
            ret.append(t)
        return ret

    def clone(self):
        ret = Buffer()
        ret.blocks = self.blocks.copy()
        return ret

    def calc_size(self):
        return sum([len(b) for b in self.blocks])

    def fill(self, buf):
        ret, tmp_buf, tmp_size = [], self.clone(), self.calc_size()
        for blk in buf:
            if tmp_size + len(blk) > CAPACITY:
                ret.append(tmp_buf)
                tmp_buf, tmp_size = self.clone(), self.calc_size()
            tmp_buf.blocks.append(blk)
            tmp_size += len(blk)
        ret.append(tmp_buf)
        return ret

    def insert(self, b, reverse=True):
        if not reverse:
            for index in range(len(self.blocks) + 1):
                if index >= len(self.blocks) or b < self.blocks[index]:
                    self.blocks.insert(index, b)
                    break
        else:
            for index in range(len(self.blocks), -1, -1):
                if index == 0 or self.blocks[index - 1] < b:
                    self.blocks.insert(index, b)
                    break

    def random_sample(self, size):
        assert size <= len(self.blocks)
        index = sorted(random.sample(range(len(self.blocks)), size))
        ret = Buffer()
        ret.blocks = [self.blocks[i] for i in index]
        return ret

    def filtered(self, fltr: 'function blk, index->bool', need_residue=False):
        """
        过滤出正负block
        :param fltr:
        :param need_residue:
        :return:
        """
        ret, ret2 = Buffer(), Buffer()
        for i, blk in enumerate(self.blocks):
            if fltr(blk, i):
                ret.blocks.append(blk)
            else:
                ret2.blocks.append(blk)
        if need_residue:
            return ret, ret2
        else:
            return ret

    def export(self):
        """计算bert的三个输入ids，att_masks，type_ids
        :return:
        """
        ids = [[]]
        att_masks = [[]]
        type_ids = [[]]
        for b in self.blocks:
            ids = np.concatenate((ids, [b.ids]), axis=1)
            att_masks = np.concatenate((att_masks, [np.ones_like(b.ids)]), axis=1)
            if b.blk_type == 1:
                type_ids = np.concatenate((type_ids, [np.array([1] * len(b))]), axis=1)
            else:
                type_ids = np.concatenate((type_ids, [np.array([0] * len(b))]), axis=1)
        inputs = np.concatenate((ids, att_masks, type_ids), axis=0)
        return inputs.astype(np.int64)

    def export_relevance(self):
        """得出block对应label
        relevance>=1=>label=1
        relevance< 1=>label=0
        """
        relevance = []
        for b in self.blocks:
            if b.relevance >= 1:
                relevance.extend([1] * len(b))
            else:
                relevance.extend([0] * len(b))
        return np.array([relevance])
