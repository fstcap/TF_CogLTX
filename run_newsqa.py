import os
import json
import tensorflow as tf
from transformers import RobertaTokenizer
from main_loop import main_loop, prediction
from utils import DEFAULT_MODEL_NAME

if __name__ == "__main__":
    main_loop()  # 训练

    tokenizer = RobertaTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    ans = {}
    for qbuf, dbuf, buf, relevance_score, ids, output in prediction(): # 预测
        _id = qbuf[0]._id
        # print(f"\033[0;35m _id:\033[0;36m{_id}\033[0m")
        # print(f"\033[0;35m ids:\033[0;36m{tf.shape(ids)}\033[0m")
        # print(f"\033[0;35m output:\033[0;36m{tf.shape(output)}\033[0m")
        start_end = tf.math.argmax(output, axis=-1)
        start_end = start_end[:]
        start = start_end[0][0]
        end = start_end[0][1] + 1

        ans_ids = ids[0][start: end]
        ans[_id] = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(ans_ids)).replace('</s>', '').replace('<pad>', '').strip()

    with open(os.path.join('tmp_dir', 'pred.json'), 'w') as fout:
        json.dump(ans, fout)
