import os
import numpy as np
import tensorflow as tf

DEFAULT_MODEL_NAME = 'roberta-base'
BLOCK_SIZE = 63  # 为什么每块长度是63因为算上sep_token是64
CAPACITY = 512  # roberta输入限制最大长度
ESTIMATIONS_FILE_PATH = os.path.join('tmp_dir', 'estimations.txt')


def score_blocks(qbuf, relevance_token):
    """计算每一个block中token得分的平均值
    :param qbuf:
    :param relevance_token:
    :return: 
    """
    ends = qbuf.block_ends()
    relevance_blk = np.ones(len(ends))
    for i in range(len(ends)):
        if qbuf[i].blk_type > 0:  # blk_type>0是句子的block
            relevance_token_block = relevance_token[ends[i - 1]:ends[i]]
            relevance_blk[i] = tf.math.reduce_mean(relevance_token_block).numpy()
    return relevance_blk
