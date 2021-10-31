import os
import numpy as np
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub

DEFAULT_MODEL_NAME = 'roberta-base'
BLOCK_SIZE = 63  # 为什么每块长度是63因为算上sep_token是64
CAPACITY = 512  # roberta输入限制最大长度
ESTIMATIONS_FILE_PATH = os.path.join('tmp_dir', 'estimations.txt')
save_dir = os.path.join('tmp_dir', 'estimations.txt')
SAVEDIR = 'save_dir'


class PreprocessorTokenize:
    def __init__(self):
        preprocessor = hub.load("https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1")
        self.start_of_sequence_id = preprocessor.tokenize.get_special_tokens_dict()['start_of_sequence_id']
        self.end_of_segment_id = preprocessor.tokenize.get_special_tokens_dict()['end_of_segment_id']
        self.hub_tokenize = hub.KerasLayer(preprocessor.tokenize)

    def tokenize(self, text):
        tokenize = self.hub_tokenize([text])
        return tokenize.to_list()[0]

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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """自定义变化学习率
    """

    def __init__(self, initial_learning_rate=2e-5, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5) + self.initial_learning_rate

        return tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,

        }
        return config
