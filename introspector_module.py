import os
import numpy as np
import tensorflow as tf
from models import Introspector
from utils import ESTIMATIONS_FILE_PATH, DEFAULT_MODEL_NAME, score_blocks, CustomSchedule, SAVEDIR


class IntrospectorModule:
    """设置分类训练的超参数和步骤
    """

    def __init__(self, train_data_size, epochs=3, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size

        steps_per_epoch = int(train_data_size / batch_size)
        num_train_steps = steps_per_epoch * epochs
        warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

        learning_rate = CustomSchedule(2e-5)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')

    def buf_to_input(self, buf):
        """
        转换成bert输入
        :param buf:
        :return:
        """
        input = buf.export()
        label = buf.export_relevance()
        return np.concatenate((input, label), axis=0)

    def train_step(self, input):
        """tf_function方法的训练步骤
        :param input:
        :return:
        """
        input_ids = input[:, 0, :]
        attention_mask = input[:, 1, :]
        y_real = input[:, 3, :][:, :, tf.newaxis]
        with tf.GradientTape() as tape:
            y_pred = self.introspector({
                'input_ids': input_ids, 'attention_mask': attention_mask})
            loss = self.loss_fn(y_real, y_pred)
        gradients = tape.gradient(loss, self.introspector.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.introspector.trainable_variables))
        return loss

    def training_epoch(self, inputs):
        """tf_function训练的epoch便利
        :param inputs:
        :return:
        """
        datasetes = tf.data.Dataset.from_tensor_slices(inputs).batch(self.batch_size)
        for epoch in range(self.epochs):
            for index, dataset in enumerate(datasetes):
                print(f"\033[0;35mIndex:\033[0;36m{index}\033[0m")
                loss = self.train_step(dataset)
                print(f"\033[0;33mLoss:\033[0;34m{loss}\033[0m")

    def _write_estimation(self, bufs, logits):
        """将每个block中每个token的得分写入estimations.txt
        :param bufs:
        :param logits:
        :return:
        """
        if not os.path.exists('tmp_dir'):
            os.makedirs('tmp_dir')

        with open(ESTIMATIONS_FILE_PATH, 'w') as f:
            for i_bufs, buf in enumerate(bufs):
                relevance_blk = score_blocks(buf, tf.math.sigmoid(logits[i_bufs]))
                for i_buf, blk in enumerate(buf):
                    f.write(f'{blk.pos} {relevance_blk[i_buf].item()}\n')

    def training_fit(self, bufs):
        """fit训练方式
        """
        inputs = [self.buf_to_input(buf) for buf in bufs]
        inputs = tf.ragged.constant(inputs).to_tensor()

        input_ids = inputs[:, 0, :]
        attention_mask = inputs[:, 1, :]
        label = inputs[:, 3, :][:, :, tf.newaxis]

        print(f"\033[0;35m input_ids:\033[0;36m{input_ids}\033[0m")
        print(f"\033[0;35m Attention_mask:\033[0;36m{attention_mask}\033[0m")

        self.introspector = Introspector.from_pretrained(DEFAULT_MODEL_NAME)
        self.introspector.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy'])

        self.introspector.fit(
            x={'input_ids': input_ids, 'attention_mask': attention_mask},
            y=label, batch_size=self.batch_size, epochs=self.epochs)

        if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)
        self.introspector.save_pretrained(os.path.join(SAVEDIR, 'introspector'))

        logits = self.introspector.predict(
            x={'input_ids': input_ids, 'attention_mask': attention_mask},
            batch_size=self.batch_size)

        self._write_estimation(bufs, logits)
