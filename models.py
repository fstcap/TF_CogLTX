import numpy as np
import tensorflow as tf
from transformers import TFRobertaMainLayer, TFRobertaPreTrainedModel


class Introspector(TFRobertaPreTrainedModel):
    """分类模型
    """

    def __init__(self, config):
        super(Introspector, self).__init__(config)

        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        self.dropout = tf.keras.layers.Dropout(rate=0.1)
        self.classifier = tf.keras.layers.Dense(1)

    def call(self,
             input_ids=None,
             attention_mask=None,
             token_type_ids=None,
             training=False):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, training=training)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

    def get_config(self):
        config = super(Introspector, self).get_config()
        return config


class QAReasoner(TFRobertaPreTrainedModel):
    """抽取式问答模型
    """

    def __init__(self, config):
        super(QAReasoner, self).__init__(config)

        self.roberta = TFRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        self.qa_outputs = tf.keras.layers.Dense(2)

    def call(self,
             input_ids=None,
             attention_mask=None,
             token_type_ids=None,
             training=False):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids, training=training)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        logits_shape = logits.shape
        start_end = tf.reshape(logits, shape=(logits_shape[0], logits_shape[2], logits_shape[1]))
        return start_end

    @classmethod
    def export_labels(cls, bufs):
        labels = np.zeros((len(bufs), 2))
        for i, buf in enumerate(bufs):
            t = 0
            for b in buf.blocks:
                if hasattr(b, 'start'):
                    labels[i, 0] = t + b.start[0]
                if hasattr(b, 'end'):
                    labels[i, 1] = t + b.end[0]
                t += len(b)
        return labels

    def get_config(self):
        config = super(Introspector, self).get_config()
        return config
