import tensorflow as tf

class GHMC_Loss(object):
    def __init__(self, bins=10, momentum=0):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target, mask):
        """ Args:
        :param input: [batch_size, seq_length, class_num]
        :param target: [batch_size, seq_length, class_num]
                    Binary target (0 or 1).
        :param mask: [batch_size, seq_length]
                    Mask matrix for valid predictions.
        :return: ghmc_loss
        """
        # batch_size = tf.shape(input)[0]
        edges = self.edges
        mmt = self.momentum
        weights = tf.zeros_like(input)

        # gradient length
        g = tf.abs(tf.stop_gradient(tf.nn.softmax(input)) - target)

        tot = tf.reduce_sum(mask[:, 1:])
        n = 0.0

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & (tf.expand_dims(mask[:, 1:], -1) > 0) & (target > 0)
            inds_ft = tf.where(inds, tf.ones_like(input), tf.zeros_like(input))
            num_in_bin = tf.reduce_sum(inds_ft)

            def valid_bin_fn():
                return n + 1.0, tf.where(inds, tot / num_in_bin * tf.ones_like(input), weights)

            def mmt_valid_bin_fn():
                self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                return n + 1.0, \
                    tf.where(inds, tot / self.acc_sum[i] * tf.ones_like(input), weights)

            def invalid_bin_fn():
                return n, weights

            if mmt == 0.0:
                n, weights = tf.cond(num_in_bin > 0,
                                     true_fn=valid_bin_fn,
                                     false_fn=invalid_bin_fn)
            else:
                n, weights = tf.cond(num_in_bin > 0,
                                     true_fn=mmt_valid_bin_fn,
                                     false_fn=invalid_bin_fn)

        weights = weights / n
        weights *= target
        weights = tf.reduce_sum(weights, axis=-1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=input)
        loss = weights * loss * mask[:, 1:]
        loss = tf.reduce_sum(loss) / tot
        return loss
