DEFAULT_MODEL_NAME = 'roberta-base'
BLOCK_SIZE = 63 # 为什么每块长度是63因为算上sep_token是64
CAPACITY = 512 # roberta输入限制最大长度