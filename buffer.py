from transformers import RobertaTokenizer
from utils import BLOCK_SIZE, DEFAULT_MODEL_NAME # 默认模型roberta-base

class Block:
    tokenizer = RobertaTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    def __init__(self, ids, pos, blk_type=1, **kwargs):
        self.ids = ids
        self.pos = pos
        self.blk_type = blk_type # 0 sentence A, 1 sentence B
        self.relevance = 0
        self.estimation = 0
        self.__dict__.update(kwargs)
    def __lt__(self, rhs):
        return self.blk_type < rhs.blk_type or (self.blk_type == rhs.blk_type and self.pos < rhs.pos)
    def __ne__(self, rhs):
        return self.pos != rhs.pos or self.blk_type != rhs.blk_type
    def __len__(self):
        return len(self.ids)
    def __str__(self):
        return Block.tokenizer.convert_tokens_to_string(Block.tokenizer.convert_ids_to_tokens(self.ids))

class Buffer:
    @staticmethod
    def split_document_into_blocks(d, tokenizer, cnt=0, hard=True, properties=None):
        '''
            d: [['word', '##piece'], ...] # a document of tokenized sentences
            properties: [
                            [
                                (name: str, value: any), # len(2) tuple, sentence level property
                                (name: str, position: int, value: any) # len(3) tuple, token level property
                            ],
                            []... # len(d) lists
                        ]
        '''
        ret = Buffer()
        updiv = lambda a, b: (a - 1) // b + 1 # 计算分组后的组数
        if hard:
            for sid, tsen in enumerate(d):
                psen = properties[sid] if properties is not None else []
                num = updiv(len(tsen), BLOCK_SIZE)  # cls
                bsize = updiv(len(tsen), num)
                for i in range(num):
                    st, en = i * bsize, min((i + 1) * bsize, len(tsen))
                    cnt += 1
                    tmp = tsen[st: en] + [tokenizer.sep_token]
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
                    ret.insert(Block(tokenizer.convert_tokens_to_ids(tmp), cnt, **tmp_kwargs))
        else:
            # d is only a list of tokens, not split.
            # properties are also a list of tuples.
            end_tokens = {'\n': 0, '.': 1, '?': 1, '!': 1, ',': 2}
            for k, v in list(end_tokens.items()):
                end_tokens['Ġ' + k] = v
            sen_cost, break_cost = 4, 8
            poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
            poses.insert(0, (-1, 0))
            if poses[-1][0] < len(d) - 1:
                poses.append((len(d) - 1, 0))
            x = 0
            while x < len(poses) - 1:
                if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
                    poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
                x += 1
            # simple dynamic programming
            best = [(0, 0)]
            for i, (p, cost) in enumerate(poses):
                if i == 0:
                    continue
                best.append((-1, 100000))
                for j in range(i - 1, -1, -1):
                    if p - poses[j][0] > BLOCK_SIZE:
                        break
                    value = best[j][1] + cost + sen_cost
                    if value < best[i][1]:
                        best[i] = (j, value)
                assert best[i][0] >= 0
            intervals, x = [], len(poses) - 1
            while x > 0:
                l = poses[best[x][0]][0]
                intervals.append((l + 1, poses[x][0] + 1))
                x = best[x][0]
            if properties is None:
                properties = []
            for st, en in reversed(intervals):
                # copy from hard version
                cnt += 1
                tmp = d[st: en] + [tokenizer.sep_token]
                # inject properties into blks
                tmp_kwargs = {}
                for p in properties:
                    if len(p) == 2:
                        tmp_kwargs[p[0]] = p[1]
                    elif len(p) == 3:
                        if st <= p[1] < en:
                            tmp_kwargs[p[0]] = (p[1] - st, p[2])
                    else:
                        raise ValueError('Invalid property {}'.format(p))
                ret.insert(Block(tokenizer.convert_tokens_to_ids(tmp), cnt, **tmp_kwargs))

        return ret, cnt
    def __init__(self):
        self.blocks = []

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, key):
        return self.blocks[key]

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