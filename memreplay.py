import tensorflow as tf
from utils import score_blocks, CAPACITY
from buffer import Buffer


def mem_replay(introspector, qbuf, dbuf, times='3,5', batch_size_inference=16):
    '''对每一个问题对应的每段的block进行打分，挑出前几个凑出小于512长度再次进行打分，获取前num_to_keep个block
    :param introspector:
    :param qbuf:
    :param dbuf:
    :param times:
    :param batch_size_inference:
    :return:
    '''
    times = [int(x) for x in times.split(',')]
    B_set = []
    for k, inc in enumerate(times):
        num_to_keep = len(qbuf) + inc

        bufs, t = qbuf.fill(dbuf), 0
        inputs = [buf.export() for buf in bufs]
        inputs = tf.ragged.constant(inputs).to_tensor()

        input_ids = inputs[:, 0, :]
        attention_mask = inputs[:, 1, :]

        logits = introspector.predict(
            x={'input_ids': input_ids, 'attention_mask': attention_mask},
            batch_size=batch_size_inference)
        estimations = []
        for i, buf in enumerate(bufs):
            estimation = score_blocks(buf, tf.math.sigmoid(logits[i]))[len(qbuf):]
            estimations.extend(estimation)
        assert len(dbuf) == len(estimations)

        indices = tf.argsort(estimations, direction='DESCENDING')
        qbuf_size = qbuf.calc_size()

        for idx in indices:
            if qbuf_size + len(dbuf[idx]) > CAPACITY:
                break
            if dbuf[idx] in B_set:
                continue
            qbuf_size += len(dbuf[idx])
            qbuf.insert(dbuf[idx])

        inputs = qbuf.export()
        inputs = tf.constant(inputs)[tf.newaxis, :, :]

        relevance_token = tf.math.sigmoid(
            introspector.predict(
                x={'input_ids': inputs[:, 0, :], 'attention_mask': inputs[:, 1, :]})[0])
        relevance_blk = score_blocks(qbuf, relevance_token)
        keeped_indices = tf.argsort(relevance_blk, direction='DESCENDING')

        if len(keeped_indices) > num_to_keep and k < len(times) - 1:
            keeped_indices = keeped_indices[:num_to_keep]
        else:
            return qbuf, relevance_blk

        filtered_qbuf, filtered_relevance_blk = Buffer(), []
        for i, blk in enumerate(qbuf):
            if i in keeped_indices:
                filtered_qbuf.blocks.append(blk)
                filtered_relevance_blk.append(relevance_blk[i])
        qbuf = filtered_qbuf
        # record the blocks already in the qbuf
        B_set = [blk for blk in qbuf if blk.blk_type == 1]

    return filtered_qbuf, tf.constant(filtered_relevance_blk)
