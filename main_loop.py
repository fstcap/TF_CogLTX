import os
import numpy as np
import tensorflow as tf
from data_helper import SimpleListDataset, BlkPosInterface
from introspector_module import IntrospectorModule
root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def buf_to_input(buf):
    """
    转换成bert输入
    :param buf:
    :return:
    """
    input = buf.export()
    label = buf.export_relevance()
    return np.concatenate((input, label), axis=0)

def main_loop():
    train_source = os.path.join(root_dir, 'data', 'newsqa_train_roberta-base.pkl')
    qd_dataset = SimpleListDataset(train_source)
    interface = BlkPosInterface(qd_dataset)

    num_samples = '1,1,1,1'
    intro_dataset = interface.build_random_buffer(num_samples=num_samples)

    inputs = [buf_to_input(buf) for buf in intro_dataset]
    tensor = tf.ragged.constant(inputs).to_tensor()
    introspector = IntrospectorModule(input_shape=tensor.shape, epochs=3, batch_size=2)
    introspector.training_fit(tensor)

if __name__ == "__main__":
    main_loop()