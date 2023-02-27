#! -*- coding: utf-8 -*-

import tensorflow as tf

'''
inputs是一个形如(batch_size, seq_len, word_size)的张量；
函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
'''
def Position_Embedding(inputs, position_size):
    batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000., \
                             2 * tf.range(position_size / 2, dtype=tf.float32 \
                            ) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    position_embedding = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_size))
    return position_embedding


'''
inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
'''
def Mask_(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12
        
def Mask(inputs, mask, mode='mul'):
    if mask == None:
        return inputs
    else:
        #mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        #for _ in range(len(inputs.shape)-2):
        #    mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12

'''
普通的全连接
inputs是一个二阶或二阶以上的张量，即形如(batch_size,...,input_size)。
只对最后一个维度做矩阵乘法，即输出一个形如(batch_size,...,ouput_size)的张量。
'''
def Dense(inputs, ouput_size, bias=True, seq_len=None):
    input_size = 512
    W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, \
                         tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                        )
    if seq_len != None:
        outputs = Mask(outputs, seq_len, 'mul')
    return outputs


def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [None, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = tf.shape(x)[-1]
    seq_len = tf.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = tf.pad(x, [[0, 0], [p_left, p_right], [0, 0]])
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = tf.concat(xs, 2)
    return tf.reshape(x, [-1, seq_len, kernel_size, seq_dim])



'''
Multi-Head Attention的实现
'''
def Attention(Q, K, V, nb_head, size_per_head, mask_Q, mask_V):
    #对Q、K、V分别作线性映射
    
    Q = Dense(Q, nb_head * size_per_head, False)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
    Q = tf.transpose(Q, [0, 2, 1, 3])
    K = Dense(K, nb_head * size_per_head, False)
    K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    V = Dense(V, nb_head * size_per_head, False)
    V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    #计算内积，然后mask，然后softmax
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, mask_V, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    #输出并mask
    O = tf.matmul(A, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    O = Mask(O, mask_Q, 'mul')
    return O

def SelfAttention(x, nb_head, size_per_head, mask_x):
    
    o = Attention(x, x, x, nb_head, size_per_head, mask_x, mask_x)
    
    return o

def AtrousSelfAttention(x, nb_head, size_per_head, rate, x_mask):
    
        
        seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = tf.shape(x)[1]
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0],[0, pad_len], [0, 0]])
        
        x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        new_seq_len = tf.shape(x)[1]
        # 变换shape
        x = tf.reshape(x, [-1, new_seq_len // rate, rate, seq_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, new_seq_len // rate, seq_dim])
        
        x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1])
        x_mask = tf.transpose(x_mask, [0, 2, 1, 3])
        x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, 1])
        # 做attention
        
        x = Attention(x, x, x, nb_head, size_per_head, x_mask, x_mask)
        out_dim = nb_head * size_per_head
        
        # 恢复shape
        x = tf.reshape(x, [-1, rate, new_seq_len // rate, out_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, new_seq_len, out_dim])
        x = x[:, : - pad_len]
        return x

def LocalSelfAttention(x, nb_head, size_per_head, neighbors, rate, x_mask):
    
    
        
        kernel_size = 1 + 2 * neighbors
        xp = extract_seq_patches(x, kernel_size, rate)
        
        
        xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 变换shape
        seq_len = tf.shape(x)[1]
        seq_dim = tf.shape(x)[-1]
        x = tf.reshape(x, [-1, 1, seq_dim])
        x_mask_new = tf.reshape(x_mask, [-1, 1, 1])
        
        xp = tf.reshape(xp, [-1, kernel_size, seq_dim])
        
        xp_mask = tf.reshape(xp_mask, [-1, kernel_size, 1])
        # 做attention
        #if x_mask is not None:
        #    x = self.reuse(self.attention, [x, xp, xp, xp_mask])
        #else:
        #    x = self.reuse(self.attention, [x, xp, xp])
        x = Attention(x, xp, xp, nb_head, size_per_head, x_mask_new, xp_mask)
        out_dim = nb_head * size_per_head
        
        # 恢复shape
        x = tf.reshape(x, [-1, seq_len, out_dim])
        #x = to_mask(x, x_mask, 'mul')
        return x    
    
    
def SparseSelfAttention(x, nb_head, size_per_head, neighbors, rate, x_mask):
    
        #neighbors = rate - 1
        
        
        seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = tf.shape(x)[1]
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        new_seq_len = tf.shape(x)[1]
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = Dense(x, out_dim, False)
        kw = Dense(x, out_dim, False)
        vw = Dense(x, out_dim, False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        a = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        a = tf.transpose(a, [0, 1, 2, 4, 3])
        a = Mask(a, x_mask, 'add')
        a = tf.transpose(a, [0, 1, 2, 4, 3])
        
        
        #if self.mask_right:
        #    ones = K.ones_like(a[: 1, : 1, : 1])
        #    mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
        #    a = a - mask
        
        
        # Attention2
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        #if x_mask is not None:
        ap = Mask(ap, xp_mask, 'add')
        ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        
        
        #if self.mask_right:
        #    mask = np.ones((1, kernel_size))
        #    mask[:, - self.neighbors : ] = 0
        #    mask = (1 - K.constant(mask)) * 1e10
        #    for _ in range(4):
        #        mask = K.expand_dims(mask, 0)
        #    ap = ap - mask
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        o1 = tf.matmul(a, vw)
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o    
    
def SparseSelfAttention_(x, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(x, out_dim, use_bias=False)
        vw = tf.layers.dense(x, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        a = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        a = tf.transpose(a, [0, 1, 2, 4, 3])
        #a = Mask(a, x_mask, 'add')
        a = tf.transpose(a, [0, 1, 2, 4, 3])
        
        
        #if self.mask_right:
        #    ones = K.ones_like(a[: 1, : 1, : 1])
        #    mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
        #    a = a - mask
        
        
        # Attention2
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        #if x_mask is not None:
        #ap = Mask(ap, xp_mask, 'add')
        ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        
        
        #if self.mask_right:
        #    mask = np.ones((1, kernel_size))
        #    mask[:, - self.neighbors : ] = 0
        #    mask = (1 - K.constant(mask)) * 1e10
        #    for _ in range(4):
        #        mask = K.expand_dims(mask, 0)
        #    ap = ap - mask
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        o1 = tf.matmul(a, vw)
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o     
    
    
def SparseCrossAttention_(x, y, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(y, out_dim, use_bias=False)
        vw = tf.layers.dense(y, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        a = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        a = tf.transpose(a, [0, 1, 2, 4, 3])
        #a = Mask(a, x_mask, 'add')
        a = tf.transpose(a, [0, 1, 2, 4, 3])
        
        
        #if self.mask_right:
        #    ones = K.ones_like(a[: 1, : 1, : 1])
        #    mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
        #    a = a - mask
        
        
        # Attention2
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        #if x_mask is not None:
        #ap = Mask(ap, xp_mask, 'add')
        ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        
        
        #if self.mask_right:
        #    mask = np.ones((1, kernel_size))
        #    mask[:, - self.neighbors : ] = 0
        #    mask = (1 - K.constant(mask)) * 1e10
        #    for _ in range(4):
        #        mask = K.expand_dims(mask, 0)
        #    ap = ap - mask
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        o1 = tf.matmul(a, vw)
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o     
    

def SparseCrossAttention_gate(x, y, glob, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(y, out_dim, use_bias=False)
        
        glob_ = tf.reshape(glob, [-1, 1, seq_dim])
        kw = kw * glob_
        
        vw = tf.layers.dense(y, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        a = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        a = tf.transpose(a, [0, 1, 2, 4, 3])
        #a = Mask(a, x_mask, 'add')
        a = tf.transpose(a, [0, 1, 2, 4, 3])
        
        
        #if self.mask_right:
        #    ones = K.ones_like(a[: 1, : 1, : 1])
        #    mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
        #    a = a - mask
        
        
        # Attention2
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        #if x_mask is not None:
        #ap = Mask(ap, xp_mask, 'add')
        ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        
        
        #if self.mask_right:
        #    mask = np.ones((1, kernel_size))
        #    mask[:, - self.neighbors : ] = 0
        #    mask = (1 - K.constant(mask)) * 1e10
        #    for _ in range(4):
        #        mask = K.expand_dims(mask, 0)
        #    ap = ap - mask
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        o1 = tf.matmul(a, vw)
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o     

    
def SparseMaxPoolAttention_(x, y, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        batch_size = tf.shape(x)[0]
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(y, out_dim, use_bias=False)
        vw = tf.layers.dense(y, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        #vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        #vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 2, 1, 3]) # shape=[None, heads, seq_len, size]
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        
        qw = tf.reshape(qw, [-1, nb_head, new_seq_len, size_per_head])
        kw = tf.reshape(kw, [-1, nb_head, new_seq_len, size_per_head])
        a_ori = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        a_ori = tf.reshape(a_ori, [-1, new_seq_len, new_seq_len // rate, rate])
        index_a = tf.argmax(a_ori, axis=3)
        max_a = tf.reduce_max(a_ori, axis=3)
        a = tf.reshape(max_a, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        
        
        index_a = tf.reshape(index_a, [-1, 1])
        indices_a = tf.expand_dims(tf.range(0, batch_size*new_seq_len*8*new_seq_len/rate, 1), 1)
        
        index_a = tf.cast(index_a, tf.int32)
        concated_a = tf.concat([indices_a, index_a], 1)
        onehot_a = tf.sparse_to_dense(concated_a, tf.stack([batch_size*new_seq_len*8*new_seq_len/rate, rate]), 1.0, 0.0)
        
        onehot_a = tf.reshape(onehot_a, [-1, new_seq_len, new_seq_len/rate, rate])
        
        
         
        #max_a = max_a * onehot_a
        #max_a = tf.reshape(max_a, [-1, new_seq_len, new_seq_len])
        #a = tf.reshape(max_a, [-1, nb_head, rate, new_seq_len/rate, new_seq_len])
        
        
        
        #a = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        #a = tf.transpose(a, [0, 1, 2, 4, 3])
        #a = Mask(a, x_mask, 'add')
        #a = tf.transpose(a, [0, 1, 2, 4, 3])
        
        
        #if self.mask_right:
        #    ones = K.ones_like(a[: 1, : 1, : 1])
        #    mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
        #    a = a - mask
        
        
        # Attention2
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        #ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        #if x_mask is not None:
        #ap = Mask(ap, xp_mask, 'add')
        #ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        
        
        #if self.mask_right:
        #    mask = np.ones((1, kernel_size))
        #    mask[:, - self.neighbors : ] = 0
        #    mask = (1 - K.constant(mask)) * 1e10
        #    for _ in range(4):
        #        mask = K.expand_dims(mask, 0)
        #    ap = ap - mask
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        
        a = tf.reshape(a, [-1, new_seq_len, new_seq_len/rate, 1])
        a = a * onehot_a
        
        a = tf.reshape(a, [-1, nb_head, new_seq_len, new_seq_len])
        
        
        o1 = tf.matmul(a, vw)
        o1 = tf.reshape(o1, [-1, nb_head, rate, new_seq_len/rate, size_per_head])
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o         
    
    
def SparseSelfMaxPoolAttention_(x, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        batch_size = tf.shape(x)[0]
        
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        #y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        #y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(x, out_dim, use_bias=False)
        vw = tf.layers.dense(x, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        #vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        #vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 2, 1, 3]) # shape=[None, heads, seq_len, size]
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        
        qw = tf.reshape(qw, [-1, nb_head, new_seq_len, size_per_head])
        kw = tf.reshape(kw, [-1, nb_head, new_seq_len, size_per_head])
        a_ori = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        a_ori = tf.reshape(a_ori, [-1, new_seq_len, new_seq_len // rate, rate])
        index_a = tf.argmax(a_ori, axis=3)
        max_a = tf.reduce_max(a_ori, axis=3)
        a = tf.reshape(max_a, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        
        
        index_a = tf.reshape(index_a, [-1, 1])
        indices_a = tf.expand_dims(tf.range(0, batch_size*new_seq_len*8*new_seq_len/rate, 1), 1)
        
        index_a = tf.cast(index_a, tf.int32)
        concated_a = tf.concat([indices_a, index_a], 1)
        onehot_a = tf.sparse_to_dense(concated_a, tf.stack([batch_size*new_seq_len*8*new_seq_len/rate, rate]), 1.0, 0.0)
        
        onehot_a = tf.reshape(onehot_a, [-1, new_seq_len, new_seq_len/rate, rate])
        
        
         
        #max_a = max_a * onehot_a
        #max_a = tf.reshape(max_a, [-1, new_seq_len, new_seq_len])
        #a = tf.reshape(max_a, [-1, nb_head, rate, new_seq_len/rate, new_seq_len])
        
        
        
        #a = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        #a = tf.transpose(a, [0, 1, 2, 4, 3])
        #a = Mask(a, x_mask, 'add')
        #a = tf.transpose(a, [0, 1, 2, 4, 3])
        
        
        #if self.mask_right:
        #    ones = K.ones_like(a[: 1, : 1, : 1])
        #    mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
        #    a = a - mask
        
        
        # Attention2
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        #ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        #if x_mask is not None:
        #ap = Mask(ap, xp_mask, 'add')
        #ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        
        
        #if self.mask_right:
        #    mask = np.ones((1, kernel_size))
        #    mask[:, - self.neighbors : ] = 0
        #    mask = (1 - K.constant(mask)) * 1e10
        #    for _ in range(4):
        #        mask = K.expand_dims(mask, 0)
        #    ap = ap - mask
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        
        a = tf.reshape(a, [-1, new_seq_len, new_seq_len/rate, 1])
        a = a * onehot_a
        
        a = tf.reshape(a, [-1, nb_head, new_seq_len, new_seq_len])
        
        
        o1 = tf.matmul(a, vw)
        o1 = tf.reshape(o1, [-1, nb_head, rate, new_seq_len/rate, size_per_head])
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o   
    
def SparseBoundaryAttention_(x, y, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        batch_size = tf.shape(x)[0]
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(y, out_dim, use_bias=False)
        vw = tf.layers.dense(y, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        #vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        #vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 2, 1, 3]) # shape=[None, heads, seq_len, size]
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        
        qw = tf.reshape(qw, [-1, nb_head, new_seq_len, size_per_head])
        kw = tf.reshape(kw, [-1, nb_head, new_seq_len, size_per_head])
        a_ori = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        
        #shape a_ori [-1, nb_head, new_seq_len, new_seq_len]
        sub_a_ori = a_ori[:,:,:,:-1]
        sub_a_ori = tf.concat([tf.expand_dims(a_ori[:,:,:,0], -1), sub_a_ori], -1)
        #shape sub_a_ori [-1, nb_head, new_seq_len, new_seq_len]
        grad = tf.abs(a_ori - sub_a_ori)
        grad = 5*grad + a_ori
        
        grad_value, grad_index = tf.nn.top_k(grad, k=new_seq_len//rate)
        
        #index [-1, nb_head, new_seq_len, grad_k]
        #vw [-1, nb_head, new_seq_len, size_per_head]
        
        grad_index = tf.reshape(grad_index, [-1, new_seq_len//rate])
        
        a_ori_ = tf.reshape(a_ori, [-1, new_seq_len])
        grad_value = tf.batch_gather(a_ori_, grad_index)
        
        temp_vw = tf.expand_dims(vw, 2)
        temp_vw = tf.tile(temp_vw, [1, 1, new_seq_len, 1, 1])
        temp_vw = tf.reshape(temp_vw, [-1, new_seq_len, size_per_head]) 
        
        grad_vw = tf.batch_gather(temp_vw, grad_index)
        #grad_vw = tf.gather(temp_vw, grad_index, axis=1)
        grad_vw = tf.reshape(grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head])
        
        
        a = tf.reshape(grad_value, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        
        
        
        #a_ori = tf.reshape(a_ori, [-1, new_seq_len, new_seq_len // rate, rate])
        #index_a = tf.argmax(a_ori, axis=3)
        #max_a = tf.reduce_max(a_ori, axis=3)
        
        
        #a = tf.reshape(max_a, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        
        
        #index_a = tf.reshape(index_a, [-1, 1])
        #indices_a = tf.expand_dims(tf.range(0, batch_size*new_seq_len*8*new_seq_len/rate, 1), 1)
        
        #index_a = tf.cast(index_a, tf.int32)
        #concated_a = tf.concat([indices_a, index_a], 1)
        #onehot_a = tf.sparse_to_dense(concated_a, tf.stack([batch_size*new_seq_len*8*new_seq_len/rate, rate]), 1.0, 0.0)
        
        #onehot_a = tf.reshape(onehot_a, [-1, new_seq_len, new_seq_len/rate, rate])
        
        
         
        #max_a = max_a * onehot_a
        #max_a = tf.reshape(max_a, [-1, new_seq_len, new_seq_len])
        #a = tf.reshape(max_a, [-1, nb_head, rate, new_seq_len/rate, new_seq_len])
        
        
        
        #a = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        #a = tf.transpose(a, [0, 1, 2, 4, 3])
        #a = Mask(a, x_mask, 'add')
        #a = tf.transpose(a, [0, 1, 2, 4, 3])
        
        
        #if self.mask_right:
        #    ones = K.ones_like(a[: 1, : 1, : 1])
        #    mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
        #    a = a - mask
        
        
        # Attention2
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        #ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        #if x_mask is not None:
        #ap = Mask(ap, xp_mask, 'add')
        #ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        
        
        #if self.mask_right:
        #    mask = np.ones((1, kernel_size))
        #    mask[:, - self.neighbors : ] = 0
        #    mask = (1 - K.constant(mask)) * 1e10
        #    for _ in range(4):
        #        mask = K.expand_dims(mask, 0)
        #    ap = ap - mask
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        
        #a = tf.reshape(a, [-1, new_seq_len, new_seq_len/rate, 1])
        #a = a * onehot_a
        
        a = tf.reshape(a, [-1, nb_head, new_seq_len, new_seq_len//rate])
        a = tf.expand_dims(a, -2)
        #grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head]
        
        
        o1 = tf.matmul(a, grad_vw)
        o1 = tf.reshape(o1, [-1, nb_head, rate, new_seq_len/rate, size_per_head])
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o         
    
def SparseSelfBoundaryAttention_(x, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        batch_size = tf.shape(x)[0]
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        #y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        #y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(x, out_dim, use_bias=False)
        vw = tf.layers.dense(x, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        #vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        #vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 2, 1, 3]) # shape=[None, heads, seq_len, size]
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        
        qw = tf.reshape(qw, [-1, nb_head, new_seq_len, size_per_head])
        kw = tf.reshape(kw, [-1, nb_head, new_seq_len, size_per_head])
        a_ori = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        
        #shape a_ori [-1, nb_head, new_seq_len, new_seq_len]
        sub_a_ori = a_ori[:,:,:,:-1]
        sub_a_ori = tf.concat([tf.expand_dims(a_ori[:,:,:,0], -1), sub_a_ori], -1)
        #shape sub_a_ori [-1, nb_head, new_seq_len, new_seq_len]
        grad = tf.abs(a_ori - sub_a_ori)
        grad = 5*grad + a_ori
        grad_value, grad_index = tf.nn.top_k(grad, k=new_seq_len//rate)
        
        #index [-1, nb_head, new_seq_len, grad_k]
        #vw [-1, nb_head, new_seq_len, size_per_head]
        
        grad_index = tf.reshape(grad_index, [-1, new_seq_len//rate])
        
        a_ori_ = tf.reshape(a_ori, [-1, new_seq_len])
        grad_value = tf.batch_gather(a_ori_, grad_index)
        
        temp_vw = tf.expand_dims(vw, 2)
        temp_vw = tf.tile(temp_vw, [1, 1, new_seq_len, 1, 1])
        temp_vw = tf.reshape(temp_vw, [-1, new_seq_len, size_per_head]) 
        
        grad_vw = tf.batch_gather(temp_vw, grad_index)
        #grad_vw = tf.gather(temp_vw, grad_index, axis=1)
        
        
        grad_vw = tf.reshape(grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head])
        
        
        a = tf.reshape(grad_value, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        
        
        
        #a_ori = tf.reshape(a_ori, [-1, new_seq_len, new_seq_len // rate, rate])
        #index_a = tf.argmax(a_ori, axis=3)
        #max_a = tf.reduce_max(a_ori, axis=3)
        
        
        #a = tf.reshape(max_a, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        
        
        #index_a = tf.reshape(index_a, [-1, 1])
        #indices_a = tf.expand_dims(tf.range(0, batch_size*new_seq_len*8*new_seq_len/rate, 1), 1)
        
        #index_a = tf.cast(index_a, tf.int32)
        #concated_a = tf.concat([indices_a, index_a], 1)
        #onehot_a = tf.sparse_to_dense(concated_a, tf.stack([batch_size*new_seq_len*8*new_seq_len/rate, rate]), 1.0, 0.0)
        
        #onehot_a = tf.reshape(onehot_a, [-1, new_seq_len, new_seq_len/rate, rate])
        
        
         
        #max_a = max_a * onehot_a
        #max_a = tf.reshape(max_a, [-1, new_seq_len, new_seq_len])
        #a = tf.reshape(max_a, [-1, nb_head, rate, new_seq_len/rate, new_seq_len])
        
        
        
        #a = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        #a = tf.transpose(a, [0, 1, 2, 4, 3])
        #a = Mask(a, x_mask, 'add')
        #a = tf.transpose(a, [0, 1, 2, 4, 3])
        
        
        #if self.mask_right:
        #    ones = K.ones_like(a[: 1, : 1, : 1])
        #    mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
        #    a = a - mask
        
        
        # Attention2
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        #ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        #if x_mask is not None:
        #ap = Mask(ap, xp_mask, 'add')
        #ap = tf.transpose(ap, [0, 1, 2, 3, 5, 4])
        
        
        #if self.mask_right:
        #    mask = np.ones((1, kernel_size))
        #    mask[:, - self.neighbors : ] = 0
        #    mask = (1 - K.constant(mask)) * 1e10
        #    for _ in range(4):
        #        mask = K.expand_dims(mask, 0)
        #    ap = ap - mask
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        
        #a = tf.reshape(a, [-1, new_seq_len, new_seq_len/rate, 1])
        #a = a * onehot_a
        
        a = tf.reshape(a, [-1, nb_head, new_seq_len, new_seq_len//rate])
        a = tf.expand_dims(a, -2)
        #grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head]
        
        
        o1 = tf.matmul(a, grad_vw)
        o1 = tf.reshape(o1, [-1, nb_head, rate, new_seq_len/rate, size_per_head])
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o  
    
def SparseGlobalAttention_(x, y, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        batch_size = tf.shape(x)[0]
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(y, out_dim, use_bias=False)
        vw = tf.layers.dense(y, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        #vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        #vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 2, 1, 3]) # shape=[None, heads, seq_len, size]
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        
        qw = tf.reshape(qw, [-1, nb_head, new_seq_len, size_per_head])
        kw = tf.reshape(kw, [-1, nb_head, new_seq_len, size_per_head])
        a_ori = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        
        #shape a_ori [-1, nb_head, new_seq_len, new_seq_len]
        sub_a_ori = a_ori[:,:,:,:-1]
        sub_a_ori = tf.concat([tf.expand_dims(a_ori[:,:,:,0], -1), sub_a_ori], -1)
        #shape sub_a_ori [-1, nb_head, new_seq_len, new_seq_len]
        grad = tf.abs(a_ori - sub_a_ori)
        grad = 5*grad + a_ori
        
        grad_value, grad_index = tf.nn.top_k(grad, k=new_seq_len//rate)
        
        #index [-1, nb_head, new_seq_len, grad_k]
        #vw [-1, nb_head, new_seq_len, size_per_head]
        
        grad_index = tf.reshape(grad_index, [-1, new_seq_len//rate])
        
        a_ori_ = tf.reshape(a_ori, [-1, new_seq_len])
        grad_value = tf.batch_gather(a_ori_, grad_index)
        
        temp_vw = tf.expand_dims(vw, 2)
        temp_vw = tf.tile(temp_vw, [1, 1, new_seq_len, 1, 1])
        temp_vw = tf.reshape(temp_vw, [-1, new_seq_len, size_per_head]) 
        
        grad_vw = tf.batch_gather(temp_vw, grad_index)
        #grad_vw = tf.gather(temp_vw, grad_index, axis=1)
        grad_vw = tf.reshape(grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head])
        
        
        a = tf.reshape(grad_value, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        
        
        
        
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        
        #a = tf.reshape(a, [-1, new_seq_len, new_seq_len/rate, 1])
        #a = a * onehot_a
        
        a = tf.reshape(a, [-1, nb_head, new_seq_len, new_seq_len//rate])
        a = tf.expand_dims(a, -2)
        #grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head]
        
        
        o1 = tf.matmul(a, grad_vw)
        o1 = tf.reshape(o1, [-1, nb_head, rate, new_seq_len/rate, size_per_head])
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 #+ o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o         

def SparseSelfGlobalAttention_(x, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        batch_size = tf.shape(x)[0]
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        #y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        #y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(x, out_dim, use_bias=False)
        vw = tf.layers.dense(x, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        #vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        #vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 2, 1, 3]) # shape=[None, heads, seq_len, size]
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        
        qw = tf.reshape(qw, [-1, nb_head, new_seq_len, size_per_head])
        kw = tf.reshape(kw, [-1, nb_head, new_seq_len, size_per_head])
        a_ori = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        
        #shape a_ori [-1, nb_head, new_seq_len, new_seq_len]
        sub_a_ori = a_ori[:,:,:,:-1]
        sub_a_ori = tf.concat([tf.expand_dims(a_ori[:,:,:,0], -1), sub_a_ori], -1)
        #shape sub_a_ori [-1, nb_head, new_seq_len, new_seq_len]
        grad = tf.abs(a_ori - sub_a_ori)
        grad = 5*grad + a_ori
        grad_value, grad_index = tf.nn.top_k(grad, k=new_seq_len//rate)
        
        #index [-1, nb_head, new_seq_len, grad_k]
        #vw [-1, nb_head, new_seq_len, size_per_head]
        
        grad_index = tf.reshape(grad_index, [-1, new_seq_len//rate])
        
        a_ori_ = tf.reshape(a_ori, [-1, new_seq_len])
        grad_value = tf.batch_gather(a_ori_, grad_index)
        
        temp_vw = tf.expand_dims(vw, 2)
        temp_vw = tf.tile(temp_vw, [1, 1, new_seq_len, 1, 1])
        temp_vw = tf.reshape(temp_vw, [-1, new_seq_len, size_per_head]) 
        
        grad_vw = tf.batch_gather(temp_vw, grad_index)
        #grad_vw = tf.gather(temp_vw, grad_index, axis=1)
        
        
        grad_vw = tf.reshape(grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head])
        
        
        a = tf.reshape(grad_value, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        
        
        
        
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        
        
        
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        
        #a = tf.reshape(a, [-1, new_seq_len, new_seq_len/rate, 1])
        #a = a * onehot_a
        
        a = tf.reshape(a, [-1, nb_head, new_seq_len, new_seq_len//rate])
        a = tf.expand_dims(a, -2)
        #grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head]
        
        
        o1 = tf.matmul(a, grad_vw)
        o1 = tf.reshape(o1, [-1, nb_head, rate, new_seq_len/rate, size_per_head])
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 #+ o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o  
    
def SparseSelfBoundaryAttention_Visual(x, nb_head, size_per_head, neighbors, rate):
    
        #neighbors = rate - 1
        
        batch_size = tf.shape(x)[0]
        
        #seq_dim = tf.shape(x)[-1]
        # 补足长度，保证可以reshape
        #seq_len = tf.shape(x)[1]
        seq_dim = 512
        seq_len = 80
        
        pad_len = rate - seq_len % rate
        x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        #y = tf.pad(y, [[0, 0], [0, pad_len], [0, 0]])
        #if x_mask is not None:
        #x_mask = tf.pad(x_mask, [[0, 0], [0, pad_len], [0, 0]])
        #new_seq_len = tf.shape(x)[1]
        new_seq_len = seq_len + pad_len
        
        x = tf.reshape(x, [-1, new_seq_len, seq_dim]) # 经过padding后shape可能变为None，所以重新声明一下shape
        #y = tf.reshape(y, [-1, new_seq_len, seq_dim])
        # 线性变换
        out_dim = nb_head * size_per_head
        qw = tf.layers.dense(x, out_dim, use_bias=False)
        kw = tf.layers.dense(x, out_dim, use_bias=False)
        vw = tf.layers.dense(x, out_dim, use_bias=False)
        # 提取局部特征
        kernel_size = 1 + 2 * neighbors
        
        kwp = extract_seq_patches(kw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, rate) # shape=[None, seq_len, kernel_size, out_dim]
        #if x_mask is not None:
        #xp_mask = extract_seq_patches(x_mask, kernel_size, rate)
        # 形状变换
        qw = tf.reshape(qw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        kw = tf.reshape(kw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        #vw = tf.reshape(vw, [-1, new_seq_len // rate, rate, nb_head, size_per_head])
        vw = tf.reshape(vw, [-1, new_seq_len, nb_head, size_per_head])
        kwp = tf.reshape(kwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        vwp = tf.reshape(vwp, [-1, new_seq_len // rate, rate, kernel_size, nb_head, size_per_head])
        #if x_mask is not None:
        #x_mask = tf.reshape(x_mask, [-1, new_seq_len // rate, rate, 1, 1])
        #xp_mask = tf.reshape(xp_mask, [-1, new_seq_len // rate, rate, kernel_size, 1, 1])
        # 维度置换
        qw = tf.transpose(qw, [0, 3, 2, 1, 4]) # shape=[None, heads, r, seq_len // r, size]
        kw = tf.transpose(kw, [0, 3, 2, 1, 4])
        #vw = tf.transpose(vw, [0, 3, 2, 1, 4])
        vw = tf.transpose(vw, [0, 2, 1, 3]) # shape=[None, heads, seq_len, size]
        qwp = tf.expand_dims(qw, 4)
        kwp = tf.transpose(kwp, [0, 4, 2, 1, 3, 5]) # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = tf.transpose(vwp, [0, 4, 2, 1, 3, 5])
        #if x_mask is not None:
        #x_mask = tf.transpose(x_mask, [0, 3, 2, 1, 4])
        #xp_mask = tf.transpose(xp_mask, [0, 4, 2, 1, 3, 5])
        # Attention1
        
        qw = tf.reshape(qw, [-1, nb_head, new_seq_len, size_per_head])
        kw = tf.reshape(kw, [-1, nb_head, new_seq_len, size_per_head])
        a_ori = tf.matmul(qw, kw, transpose_b = True) / size_per_head**0.5
        
        #shape a_ori [-1, nb_head, new_seq_len, new_seq_len]
        sub_a_ori = a_ori[:,:,:,:-1]
        sub_a_ori = tf.concat([tf.expand_dims(a_ori[:,:,:,0], -1), sub_a_ori], -1)
        #shape sub_a_ori [-1, nb_head, new_seq_len, new_seq_len]
        grad = tf.abs(a_ori - sub_a_ori)
        grad = 5*grad + a_ori
        grad_value, grad_index = tf.nn.top_k(grad, k=new_seq_len//rate)
        
        temp_index = grad_index
        temp_ori = tf.nn.softmax(a_ori, -1)
        
        #index [-1, nb_head, new_seq_len, grad_k]
        #vw [-1, nb_head, new_seq_len, size_per_head]
        
        grad_index = tf.reshape(grad_index, [-1, new_seq_len//rate])
        
        a_ori_ = tf.reshape(a_ori, [-1, new_seq_len])
        grad_value = tf.batch_gather(a_ori_, grad_index)
        
        temp_vw = tf.expand_dims(vw, 2)
        temp_vw = tf.tile(temp_vw, [1, 1, new_seq_len, 1, 1])
        temp_vw = tf.reshape(temp_vw, [-1, new_seq_len, size_per_head]) 
        
        grad_vw = tf.batch_gather(temp_vw, grad_index)
        #grad_vw = tf.gather(temp_vw, grad_index, axis=1)
        
        
        grad_vw = tf.reshape(grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head])
        
        
        a = tf.reshape(grad_value, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
        ap = tf.matmul(qwp, kwp, transpose_b = True) / size_per_head**0.5
        
        ap = ap[..., 0, :]
        
        temp_ori = tf.nn.softmax(tf.reshape(ap, [-1, 8, 85, 9]), -1)
        
        # 合并两个Attention
        A = tf.concat([a, ap], -1)
        A = tf.nn.softmax(A, -1)
        a, ap = A[..., : tf.shape(a)[-1]], A[..., tf.shape(a)[-1] : ]
        # 完成输出1
        
        #a = tf.reshape(a, [-1, new_seq_len, new_seq_len/rate, 1])
        #a = a * onehot_a
        
        a = tf.reshape(a, [-1, nb_head, new_seq_len, new_seq_len//rate])
        a = tf.expand_dims(a, -2)
        #grad_vw, [-1, nb_head, new_seq_len, new_seq_len//rate, size_per_head]
        
        
        o1 = tf.matmul(a, grad_vw)
        o1 = tf.reshape(o1, [-1, nb_head, rate, new_seq_len/rate, size_per_head])
        # 完成输出2
        ap = tf.expand_dims(ap, -2)
        o2 = tf.matmul(ap, vwp)
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        #o = Mask(o, x_mask, 'mul')
        o = tf.transpose(o, [0, 3, 2, 1, 4])
        o = tf.reshape(o, [-1, new_seq_len, out_dim])
        o = o[:, : - pad_len]
        return o, temp_ori, temp_index

    
    
    
    