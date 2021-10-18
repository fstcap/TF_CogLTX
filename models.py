import tensorflow as tf
from transformers import TFRobertaModel
from utils import DEFAULT_MODEL_NAME


class Introspector(tf.keras.Model):
    """分类模型
    """

    def __init__(self, sequence_length):
        super(Introspector, self).__init__()

        self.roberta = TFRobertaModel.from_pretrained(DEFAULT_MODEL_NAME)
        self.dropout = tf.keras.layers.Dropout(rate=0.1)
        self.classifier = tf.keras.layers.Dense(sequence_length)

    def call(self, inputs, trainable=True):
        outputs = self.roberta(inputs['input_ids'], inputs['attention_mask'], training=trainable)
        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)[:, :, tf.newaxis]
        return logits
