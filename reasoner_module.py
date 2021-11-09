import os
import tensorflow as tf
from models import QAReasoner
from utils import CustomSchedule, DEFAULT_MODEL_NAME, SAVEDIR


class ReasonerModule:
    def __init__(self, train_data_size, epochs=3, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size

        steps_per_epoch = int(train_data_size / batch_size)
        num_train_steps = steps_per_epoch * epochs
        warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

        learning_rate = CustomSchedule(2e-5, warmup_steps)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy')

        self.qa_reasoner = QAReasoner.from_pretrained(DEFAULT_MODEL_NAME)

    def buf_to_input(self, buf):
        input = buf.export()
        return input

    def train_step(self, input, label):
        input_ids = input[:, 0, :]
        attention_mask = input[:, 1, :]
        y_real = label

        with tf.GradientTape() as tape:
            y_pred = self.qa_reasoner({
                'input_ids': input_ids, 'attention_mask': attention_mask})
            loss = self.loss_fn(y_real, y_pred)
        gradients = tape.gradient(loss, self.qa_reasoner.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.qa_reasoner.trainable_variables))
        return loss

    def training_epoch(self, bufs):
        inputs = [self.buf_to_input(buf) for buf in bufs]
        inputs = tf.ragged.constant(inputs).to_tensor()
        labels = self.qa_reasoner.export_labels(bufs)
        labels = tf.constant(labels)

        datasetes = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.batch_size)
        for input, label in datasetes:
            loss = self.train_step(input, label)
            print(f"\033[0;35m Loss:\033[0;36m{loss}\033[0m")
            break

    def training_fit(self, bufs):
        inputs = [self.buf_to_input(buf) for buf in bufs]
        inputs = tf.ragged.constant(inputs).to_tensor()
        input_ids = inputs[:, 0, :]
        attention_mask = inputs[:, 1, :]

        labels = self.qa_reasoner.export_labels(bufs)
        labels = tf.constant(labels)

        self.qa_reasoner.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy'])
        self.qa_reasoner.fit(
            x={'input_ids': input_ids, 'attention_mask': attention_mask},
            y=labels, batch_size=self.batch_size, epochs=self.epochs)

        self.qa_reasoner.save_pretrained(os.path.join(SAVEDIR, 'reasoner'))
