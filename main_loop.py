import os
import tensorflow as tf

from data_helper import SimpleListDataset, BlkPosInterface
from introspector_module import IntrospectorModule
from reasoner_module import ReasonerModule
from models import Introspector, QAReasoner
from tqdm import tqdm  # 引入进度条模块
from utils import SAVEDIR, CAPACITY, BLOCK_SIZE
from memreplay import mem_replay

root_dir = os.path.abspath(os.path.dirname(__file__))
print(f"\033[0;35m root_dir:\033[0;36m{root_dir}\033[0m")


def main_loop():
    train_source = os.path.join(root_dir, 'data', 'newsqa_train_roberta-base.pkl')
    qd_dataset = SimpleListDataset(train_source)
    interface = BlkPosInterface(qd_dataset)

    num_samples = '1,1,1,1'
    intro_dataset = interface.build_random_buffer(num_samples=num_samples)
    introspector = IntrospectorModule(train_data_size=len(intro_dataset), epochs=3, batch_size=2)
    introspector.training_fit(intro_dataset)

    interface.collect_estimations_from_dir()
    reason_dataset = interface.build_promising_buffer(num_samples=num_samples)
    reasoner = ReasonerModule(train_data_size=len(reason_dataset), epochs=3, batch_size=2)
    reasoner.training_fit(reason_dataset)


def prediction():
    config_times = "3,5"

    intro_model = Introspector.from_pretrained(os.path.join(SAVEDIR, 'introspector'))
    reason_model = QAReasoner.from_pretrained(os.path.join(SAVEDIR, 'reasoner'))
    test_source = os.path.join(root_dir, 'data', 'newsqa_test_roberta-base.pkl')
    qd_dataset = SimpleListDataset(test_source)

    print(f"\033[0;35m qd_dataset len:\033[0;36m{len(qd_dataset)}\033[0m")
    for qbuf, dbuf in tqdm(qd_dataset):

        if qbuf.calc_size() > CAPACITY - BLOCK_SIZE - 1:
            continue
        buf, relevance_score = mem_replay(intro_model, qbuf, dbuf, config_times, 2)
        input = buf.export()
        input = tf.constant(input)[tf.newaxis, :, :]

        output = reason_model.predict(
            x={'input_ids': input[:, 0, :], 'attention_mask': input[:, 1, :]})
        yield qbuf, dbuf, buf, relevance_score, input[:, 0, :], output

if __name__ == "__main__":
    main_loop()
    prediction()
