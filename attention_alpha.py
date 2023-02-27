# coding: utf-8
import math
import tensorflow as tf
import numpy as np
import beam_search
import layers 
from attention_model import Attention_Model

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

from IPython import embed
from sparse_transformer_alpha import *
import common_attention as ATT
    
def conv3d_layer(x, filters, kernel_size, strides=(1, 1, 1), padding='valid', activation=None, name=None):
    # fan_in = kernel_size[0] * kernel_size[1] * kernel_size[2] * tf.shape(x)[-1]
    fan_in = kernel_size[0] * kernel_size[1] * kernel_size[2] * 832
    fan_out = kernel_size[0] * kernel_size[1] * kernel_size[2] * filters
    val = math.sqrt(6.0 / (fan_in + fan_out))
    return tf.layers.conv3d(x,
                        filters,
                        kernel_size,
                        strides=strides,
                        padding=padding,
                        kernel_initializer=tf.random_uniform_initializer(minval=-val, maxval=val),
                        activation=activation,
                        name=name)



class MV_Top_Attention_Model(Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(MV_Top_Attention_Model, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.layer_num = layer_num
        self.mce = mce
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([1536, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1024, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
        
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            encoder_decoder_pool_attention_bias)
        return decoder_output
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x, y)
                        with tf.variable_scope("encdec_attention_top"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_top = self.layer_postprocess(x_old, y)
                        with tf.variable_scope("layer_attention"):
                            x = 0.9 * x + 0.1 * x_top
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn(self, inputs, inputs_top, targets):
        encoder_output = {}
        encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + self.layer_preprocess(inputs)
            # encoder_top_output = encoder_top_output + self.layer_preprocess(inputs_top)
            inputs = self.layer_preprocess(inputs)
            inputs = tf.reshape(inputs, [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = self.layer_preprocess(inputs_top)
            inputs_top = tf.reshape(inputs_top, [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias)
        return encoder_output, decoder_output

    def weight_decay(self, decay_rate=1e-4):
        vars_list = tf.trainable_variables()
        weight_decays = []
        for v in vars_list:
            is_bias = len(v.shape.as_list()) == 1 or v.name.endswith("bias:0")
            is_emb = v.name.endswith("emb:0")
            if not (is_bias or is_emb):
                v_loss = tf.nn.l2_loss(v)
                weight_decays.append(v_loss)
        return tf.add_n(weight_decays) * decay_rate
    
    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,:1536]
        self.video_flat = self.video_flat[:,1536:]
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1) 

        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:,1:]
            # loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
            loss = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
        total_loss = loss + reg_loss
        
        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("total_loss", total_loss)
        return total_loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0.6):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size * beam_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        decode_length = tf.constant(decode_length)
        
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]    

class MV_Top_Attention_Model_Audio(Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 n_audio,
                 dim_audio=128,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(MV_Top_Attention_Model_Audio, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.layer_num = layer_num
        self.mce = mce
        self.n_audio = n_audio
        self.dim_audio = dim_audio
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([4032, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1024, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
            
        with tf.variable_scope("W_audio_encode"):
            self.encode_audio_W = tf.Variable(tf.random_normal([dim_audio, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="top_w")
            self.encode_audio_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
            
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias, encoder_decoder_audio_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            encoder_decoder_pool_attention_bias,
            encoder_decoder_audio_attention_bias)
        return decoder_output
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      encoder_decoder_audio_attention_bias,
                      name="decoder"):
        x = decoder_input
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x, y)
                        with tf.variable_scope("encdec_attention_top"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_top = self.layer_postprocess(x_old, y)
                        with tf.variable_scope("encdec_attention_audio"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_audio_output"],
                                encoder_decoder_audio_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_audio = self.layer_postprocess(x_old, y)
                        with tf.variable_scope("layer_attention"):
                            x = 0.4 * x + 0.4 * x_top + 0.2 * x_audio
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn(self, inputs, inputs_top, inputs_audio, targets):
        encoder_output = {}
        encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        encoder_audio_output, encoder_decoder_audio_attention_bias = self.encode(inputs_audio, num_encode_layers=self.encoder_layers, name="encode_audio")
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + self.layer_preprocess(inputs)
            # encoder_top_output = encoder_top_output + self.layer_preprocess(inputs_top)
            inputs = self.layer_preprocess(inputs)
            inputs = tf.reshape(inputs, [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = self.layer_preprocess(inputs_top)
            inputs_top = tf.reshape(inputs_top, [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        encoder_output["encoder_audio_output"] = encoder_audio_output
        
        decoder_output = self.decode(targets, 
                                     encoder_output, 
                                     encoder_decoder_attention_bias, 
                                     encoder_decoder_top_attention_bias, 
                                     encoder_decoder_audio_attention_bias)
        return encoder_output, decoder_output

    def weight_decay(self, decay_rate=1e-4):
        vars_list = tf.trainable_variables()
        weight_decays = []
        for v in vars_list:
            is_bias = len(v.shape.as_list()) == 1 or v.name.endswith("bias:0")
            is_emb = v.name.endswith("emb:0")
            if not (is_bias or is_emb):
                v_loss = tf.nn.l2_loss(v)
                weight_decays.append(v_loss)
        return tf.add_n(weight_decays) * decay_rate
    
    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   

        self.audio = tf.placeholder(tf.float32, [self.batch_size, self.n_audio, self.dim_audio])
        self.audio_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_audio])

        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,4032:]
        self.video_flat = self.video_flat[:,:4032]
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1) 

        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
        with tf.variable_scope("inputs_audio"):
            self.audio_flat = tf.reshape(self.audio, [-1, self.dim_audio])
            self.audio_emb = tf.nn.xw_plus_b(self.audio_flat, self.encode_audio_W, self.encode_audio_b)
            self.audio_emb = tf.reshape(self.audio_emb, [self.batch_size, self.n_audio, self.dim_model])
            self.audio_emb = self.audio_emb * tf.expand_dims(self.audio_mask, -1) 
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.audio_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:,1:]
            # loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
            loss = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
        total_loss = loss + reg_loss
        
        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("total_loss", total_loss)
        return total_loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, self.audio_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0.6):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [self.batch_size * beam_size, self.n_frames, self.dim_model])

        audio_emb = self.audio_emb
        audio_emb = tf.expand_dims(audio_emb, 1)
        audio_emb = tf.tile(audio_emb, [1, beam_size, 1, 1])
        audio_emb = tf.reshape(audio_emb, [self.batch_size * beam_size, self.n_audio, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, audio_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size * beam_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        decode_length = tf.constant(decode_length)
        
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]


class TVTAttFusionModel(MV_Top_Attention_Model_Audio):
    def __init__(self,
                 batch_size,
                 n_frames,
                 dim_image,
                 n_words,
                 vocab_size,
                 dim_model,
                 dim_hidden,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 layer_num,
                 n_audio,
                 dim_audio=128,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(TVTAttFusionModel, self).__init__(batch_size,
                                                n_frames,
                                                dim_image,
                                                n_words,
                                                vocab_size,
                                                dim_model,
                                                dim_hidden,
                                                num_heads,
                                                encoder_layers,
                                                decoder_layers,
                                                layer_num,
                                                n_audio,
                                                dim_audio,
                                                preprocess_dropout,
                                                attention_dropout,
                                                bias_init_vector,
                                                conv_before_enc,
                                                swish_activation,
                                                use_gated_linear_unit)


    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      encoder_decoder_audio_attention_bias,
                      name="decoder"):
        x = decoder_input
        current_t = tf.shape(x)[1]
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            # x = self.layer_postprocess(x, y)
                            x = y
                        with tf.variable_scope("encdec_attention_top"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            # x_top = self.layer_postprocess(x_old, y)
                            x_top = y
                        with tf.variable_scope("encdec_attention_audio"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_audio_output"],
                                encoder_decoder_audio_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            # x_audio = self.layer_postprocess(x_old, y)
                            x_audio = y
                        with tf.variable_scope("layer_attention"):
                            x_top = tf.expand_dims(x_top, 2)
                            x = tf.expand_dims(x, 2)
                            x_audio = tf.expand_dims(x_audio, 2)
                            x_cap = tf.expand_dims(x_old, 2)
                            x = tf.concat([x, x_top, x_audio, x_cap], 2)
                            # x = tf.concat([x_top, x], 2)
                            x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 3, self.dim_model]), 1)
                            # x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 2, self.dim_model]), 1)
                            x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_input),
                                x,
                                None,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x_input, y)
                            x = tf.reshape(x, [-1, current_t, self.dim_model])
                            # x = 0.8 * x + 0.2 * x_audio
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)

class Late_Fusion_Attention_Model(Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(Late_Fusion_Attention_Model, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.layer_num = layer_num
        self.mce = mce
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([4032, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1024, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
        
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output_0 = self.model_decoder(
            decoder_input,
            encoder_output["encoder_output"],
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            name="decoder_0")
        decoder_output_1 = self.model_decoder(
            decoder_input,
            encoder_output["encoder_top_output"],
            decoder_self_attention_bias,
            encoder_decoder_pool_attention_bias,
            name="decoder_1")
        return 0.5 * decoder_output_0 + 0.5 * decoder_output_1
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      name="decoder"):
        x = decoder_input
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output,
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn(self, inputs, inputs_top, targets):
        encoder_output = {}
        encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + self.layer_preprocess(inputs)
            # encoder_top_output = encoder_top_output + self.layer_preprocess(inputs_top)
            inputs = self.layer_preprocess(inputs)
            inputs = tf.reshape(inputs, [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = self.layer_preprocess(inputs_top)
            inputs_top = tf.reshape(inputs_top, [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias)
        return encoder_output, decoder_output

    def weight_decay(self, decay_rate=1e-4):
        vars_list = tf.trainable_variables()
        weight_decays = []
        for v in vars_list:
            is_bias = len(v.shape.as_list()) == 1 or v.name.endswith("bias:0")
            is_emb = v.name.endswith("emb:0")
            if not (is_bias or is_emb):
                v_loss = tf.nn.l2_loss(v)
                weight_decays.append(v_loss)
        return tf.add_n(weight_decays) * decay_rate

    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,4032:]
        self.video_flat = self.video_flat[:,:4032]
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1) 

        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:,1:]
            # loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
            loss = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
        total_loss = loss + reg_loss
        
        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("total_loss", total_loss)
        return loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0.6):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size * beam_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        decode_length = tf.constant(decode_length)
        
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]    

    
class Hierachy_Attention_Model(Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(Hierachy_Attention_Model, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.layer_num = layer_num
        self.mce = mce
        
        with tf.device("/cpu:0"):
            self.word_emb = tf.Variable(tf.random_uniform([vocab_size, 300], -0.05, 0.05), name='word_emb')
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([1836, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1324, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
        
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            encoder_decoder_pool_attention_bias)
        return decoder_output
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        current_t = tf.shape(x)[1]
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention_top"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = y
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_top = y
                        
                        
                        with tf.variable_scope("layer_attention"):
                            x_top = tf.expand_dims(x_top, 2)
                            x = tf.expand_dims(x, 2)
                            x_cap = tf.expand_dims(x_old, 2)
                            x = tf.concat([x_top, x, x_cap], 2)
                            # x = tf.concat([x_top, x], 2)
                            x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 2, self.dim_model]), 1)
                            x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_input),
                                x,
                                None,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                1,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x_input, y)
                            x = tf.reshape(x, [-1, current_t, self.dim_model])
                        
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn(self, inputs, inputs_top, targets):
        encoder_output = {}
        encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + self.layer_preprocess(inputs)
            # encoder_top_output = encoder_top_output + self.layer_preprocess(inputs_top)
            inputs = self.layer_preprocess(inputs)
            inputs = tf.reshape(inputs, [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = self.layer_preprocess(inputs_top)
            inputs_top = tf.reshape(inputs_top, [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias)
        return encoder_output, decoder_output
    
    def weight_decay(self, decay_rate=1e-4):
        vars_list = tf.trainable_variables()
        weight_decays = []
        for v in vars_list:
            is_bias = len(v.shape.as_list()) == 1 or v.name.endswith("bias:0")
            is_emb = v.name.endswith("emb:0")
            if not (is_bias or is_emb):
                v_loss = tf.nn.l2_loss(v)
                weight_decays.append(v_loss)
        return tf.add_n(weight_decays) * decay_rate
    
    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,1836:]
        self.video_flat = self.video_flat[:,:1836]
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1) 
        # self.image_emb = tf.nn.dropout(self.image_emb, keep_prob=0.5)

        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
            # self.top_emb = tf.nn.dropout(self.top_emb, keep_prob=0.5)
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        
        #print self.sent_emb.shape
        
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        # self.sent_emb = tf.nn.dropout(self.sent_emb, keep_prob=0.5)
        
        #print self.sent_emb
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        #print self.logits
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        #print self.labels
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            #print loss
            
            loss = loss * self.caption_mask[:,1:]
            
            #print loss
            # loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
            loss = tf.reduce_sum(loss) / self.batch_size
            
            #print loss
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
            #print reg_loss
        total_loss = loss + reg_loss
        #print total_loss
        
        #embed()
        
        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("total_loss", total_loss)
        return loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0.6):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size * beam_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        decode_length = tf.constant(decode_length)
        
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]    
    
    
class MV_Top_Cross_Attention_Model(Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(MV_Top_Cross_Attention_Model, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.layer_num = layer_num
        self.mce = mce
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([1536, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1024, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
        
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs[0]))
        encoder_top_input, self_attention_top_bias, encoder_decoder_attention_top_bias = (
            self.model_prepare_encoder(inputs[1]))
        encoder_output = self.model_encoder(
            encoder_input,
            encoder_top_input,
            self_attention_bias,
            self_attention_top_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias, encoder_decoder_attention_top_bias
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            encoder_decoder_pool_attention_bias)
        return decoder_output
    
    def model_encoder(self,
                      encoder_input,
                      encoder_top_input,
                      encoder_self_attention_bias,
                      encoder_self_attention_top_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        x_top = encoder_top_input
        encoder_output = {}
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("self_attention_top"):
                        # x_cross = common_layers.conv1d(x, self.dim_model, 1, name="cross_transform")
                        x_cross = tf.nn.sigmoid(common_layers.conv1d(x, self.dim_model, 1, name="cross_transform"))
                        x_top_new = x_cross * x_top
                        y_top = common_attention.multihead_attention(
                                self.layer_preprocess(x_top_new),
                                None,
                                encoder_self_attention_top_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                        x_top = self.layer_postprocess(x_top, y_top)
                    with tf.variable_scope("ffn_top"):
                        y_top = self.model_ffn_layer(self.layer_preprocess(x_top))
                        x_top = self.layer_postprocess(x_top, y_top)
            encoder_output["encoder_output"] = self.layer_preprocess(x)
            encoder_output["encoder_top_output"] = self.layer_preprocess(x_top)
        return encoder_output
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x, y)
                        with tf.variable_scope("encdec_attention_top"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_top = self.layer_postprocess(x_old, y)
                        with tf.variable_scope("layer_attention"):
                            # x_top = tf.expand_dims(x_top, 2)
                            # x_pool5 = tf.expand_dims(x_pool5, 2)
                            # x = tf.concat([x_top, x_pool5], 2)
                            # x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 1, self.dim_model]), 1)
                            # x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            # y = common_attention.multihead_attention(
                            #     self.layer_preprocess(x_input),
                            #     x,
                            #     None,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.num_heads,
                            #     self.attention_dropout,
                            #     attention_type="dot_product")
                            # x = self.layer_postprocess(x_input, y)
                            # x = tf.reshape(x, [self.batch_size, -1, self.dim_model])
                            
                            # y_pool5 = common_layers.conv1d(
                            #     self.layer_preprocess(x_pool5), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="SAME")
                            # x_pool5 = self.layer_postprocess(x_pool5, y_pool5)
                            
                            x = 0.7 * x + 0.3 * x_top
                            # x = tf.concat([x_top, x_pool5], 2)
                            # x = common_layers.conv1d(
                            #     self.layer_preprocess(x, layer_num=1), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="valid")
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn(self, inputs, inputs_top, targets):
        inputs_encode = []
        inputs_encode.append(inputs)
        inputs_encode.append(inputs_top)
        encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias = self.encode(inputs_encode, num_encode_layers=self.encoder_layers)
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + self.layer_preprocess(inputs)
            # encoder_top_output = encoder_top_output + self.layer_preprocess(inputs_top)
            inputs = self.layer_preprocess(inputs)
            inputs = tf.reshape(inputs, [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = self.layer_preprocess(inputs_top)
            inputs_top = tf.reshape(inputs_top, [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias)
        return encoder_output, decoder_output

    def weight_decay(self, decay_rate=1e-4):
        vars_list = tf.trainable_variables()
        weight_decays = []
        for v in vars_list:
            is_bias = len(v.shape.as_list()) == 1 or v.name.endswith("bias:0")
            is_emb = v.name.endswith("emb:0")
            if not (is_bias or is_emb):
                v_loss = tf.nn.l2_loss(v)
                weight_decays.append(v_loss)
        return tf.add_n(weight_decays) * decay_rate

    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,1536:]
        self.video_flat = self.video_flat[:,:1536]
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1) 

        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:,1:]
            loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
        tf.summary.scalar("training_loss", loss)
        
        return loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0.6):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size * beam_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        decode_length = tf.constant(decode_length)
        
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]    
    
class MV_Top_Object_Attention_Model(Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image, 
                 n_objects,
                 dim_object,
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(MV_Top_Object_Attention_Model, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.n_objects = n_objects
        self.dim_object = dim_object
        self.layer_num = layer_num
        self.mce = mce
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([1536, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_object_encode"):
            self.encode_object_W = tf.Variable(tf.random_normal([dim_object, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="object_w")
            self.encode_object_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="object_b")
        
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1024, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
        
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def model_object_prepare_encoder(self, inputs):
        encoder_input = inputs
        encoder_padding = common_attention.embedding_to_padding(encoder_input)
        ignore_padding = common_attention.attention_bias_ignore_padding(
            encoder_padding)
        # encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding
        # encoder_input = common_attention.add_timing_signal_1d(encoder_input)
        return (encoder_input, encoder_decoder_attention_bias)

    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias, encoder_decoder_object_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            encoder_decoder_pool_attention_bias,
            encoder_decoder_object_attention_bias)
        return decoder_output
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      encoder_decoder_object_attention_bias,
                      name="decoder"):
        x = decoder_input
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x, y)
                        with tf.variable_scope("encdec_attention_top"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_top = self.layer_postprocess(x_old, y)
                        with tf.variable_scope("encdec_attention_object"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_object_output"],
                                encoder_decoder_object_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_object = self.layer_postprocess(x_old, y)
                        with tf.variable_scope("layer_attention"):
                            # x_top = tf.expand_dims(x_top, 2)
                            # x_pool5 = tf.expand_dims(x_pool5, 2)
                            # x = tf.concat([x_top, x_pool5], 2)
                            # x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 1, self.dim_model]), 1)
                            # x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            # y = common_attention.multihead_attention(
                            #     self.layer_preprocess(x_input),
                            #     x,
                            #     None,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.num_heads,
                            #     self.attention_dropout,
                            #     attention_type="dot_product")
                            # x = self.layer_postprocess(x_input, y)
                            # x = tf.reshape(x, [self.batch_size, -1, self.dim_model])
                            
                            # y_pool5 = common_layers.conv1d(
                            #     self.layer_preprocess(x_pool5), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="SAME")
                            # x_pool5 = self.layer_postprocess(x_pool5, y_pool5)
                            
                            x = 0.4 * x + 0.4 * x_top + 0.2 * x_object
                            # x = tf.concat([x_top, x_pool5], 2)
                            # x = common_layers.conv1d(
                            #     self.layer_preprocess(x, layer_num=1), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="valid")
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn(self, inputs, inputs_top, inputs_object, targets):
        encoder_output = {}
        encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        encoder_object_output, encoder_decoder_object_attention_bias = self.model_object_prepare_encoder(inputs_object)
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + self.layer_preprocess(inputs)
            # encoder_top_output = encoder_top_output + self.layer_preprocess(inputs_top)
            inputs = self.layer_preprocess(inputs)
            inputs = tf.reshape(inputs, [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = self.layer_preprocess(inputs_top)
            inputs_top = tf.reshape(inputs_top, [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        encoder_output["encoder_object_output"] = encoder_object_output
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias, encoder_decoder_object_attention_bias)
        return encoder_output, decoder_output
    
    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   
        self.video_object = tf.placeholder(tf.float32, [self.batch_size, self.n_objects, self.dim_object])
        self.video_object_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_objects])   
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,1536:]
        self.video_flat = self.video_flat[:,:1536]
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1) 

        self.video_object_flat = tf.reshape(self.video_object, [-1, self.dim_object])
        self.object_emb = tf.nn.xw_plus_b(self.video_object_flat, self.encode_object_W, self.encode_object_b)
        self.object_emb = tf.reshape(self.object_emb, [self.batch_size, self.n_objects, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.object_emb = self.object_emb * tf.expand_dims(self.video_object_mask, -1) 
        
        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.object_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:,1:]
            loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
        tf.summary.scalar("training_loss", loss)
        
        return loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, self.object_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0.6):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        object_emb = self.object_emb
        object_emb = tf.expand_dims(object_emb, 1)
        object_emb = tf.tile(object_emb, [1, beam_size, 1, 1])
        object_emb = tf.reshape(object_emb, [self.batch_size * beam_size, self.n_objects, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, object_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size * beam_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        decode_length = tf.constant(decode_length)
        
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]    

    
class MV_Top_Scene_Attention_Model(Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image,
                 dim_scene,
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(MV_Top_Scene_Attention_Model, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.layer_num = layer_num
        self.dim_scene = dim_scene
        self.mce = mce
        
        # with tf.variable_scope("W_scene_encode"):
        #     self.encode_scene_W = tf.Variable(tf.random_normal([dim_scene, 50], 0.0, (dim_model**-0.5)), dtype=tf.float32, name="scene_w")
        #     self.encode_scene_b = tf.Variable(tf.zeros([50]), dtype=tf.float32, name="scene_b")
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([1536+self.dim_scene, dim_model], 0.0, (dim_model**-0.5)), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1024, dim_model], 0.0, (dim_model**-0.5)), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
        
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            encoder_decoder_pool_attention_bias)
        return decoder_output
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    if self.conv_before_enc == True:
                        with tf.variable_scope("conv_before"):
                            y = common_layers.conv1d(
                                self.layer_preprocess(x), 
                                self.dim_model, 
                                3, 
                                activation=tf.nn.relu, 
                                name="conv",
                                padding="SAME")
                            x = self.layer_postprocess(x, y)
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        with tf.variable_scope("multi_view"):
            encoder_output["encoder_output"] = tf.concat([encoder_output["encoder_output"], encoder_output["encoder_top_output"]], -1)
            encoder_output["encoder_output"] = common_layers.conv1d(self.layer_preprocess(encoder_output["encoder_output"], dim_feat=2*self.dim_model),
                                                                    self.dim_model,
                                                                    1)
        # current_t = tf.shape(x)[1]
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x, y)
                            
                        # with tf.variable_scope("encdec_attention_top"):
                        #     y = common_attention.multihead_attention(
                        #         self.layer_preprocess(x_old),
                        #         encoder_output["encoder_top_output"],
                        #         encoder_decoder_pool_attention_bias,
                        #         self.dim_model,
                        #         self.dim_model,
                        #         self.dim_model,
                        #         self.num_heads,
                        #         self.attention_dropout,
                        #         attention_type="dot_product")
                        #     x_top = self.layer_postprocess(x_old, y)
                            
                        # with tf.variable_scope("layer_attention"):
                            # x_top = tf.expand_dims(x_top, 2)
                            # x = tf.expand_dims(x, 2)
                            # x = tf.concat([x_top, x], 2)
                            # x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 1, self.dim_model]), 1)
                            # x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            # y = common_attention.multihead_attention(
                            #     self.layer_preprocess(x_input),
                            #     x,
                            #     None,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.num_heads,
                            #     self.attention_dropout,
                            #     attention_type="dot_product")
                            # x = self.layer_postprocess(x_input, y)
                            # x = y
                            # x = tf.reshape(x, [-1, current_t, self.dim_model])
                            
                            # y_pool5 = common_layers.conv1d(
                            #     self.layer_preprocess(x_pool5), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="SAME")
                            # x_pool5 = self.layer_postprocess(x_pool5, y_pool5)
                            # x = x + x_top
                            
                            # x = tf.concat([x_top, x_pool5], 2)
                            # x = common_layers.conv1d(
                            #     self.layer_preprocess(x, layer_num=1), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="valid")
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn(self, inputs, inputs_top, targets):
        encoder_output = {}
        encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + inputs
            # encoder_top_output = encoder_top_output + inputs_top
            inputs = tf.reshape(self.layer_preprocess(inputs), [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = tf.reshape(self.layer_preprocess(inputs_top), [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias)
        return encoder_output, decoder_output
    
    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_scene = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_scene])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   
        
        self.video_scene_flat = tf.reshape(self.video_scene, [-1, self.dim_scene])
        # self.scene_emb = tf.nn.relu(tf.nn.xw_plus_b(self.video_scene_flat, self.encode_scene_W, self.encode_scene_b))
        self.scene_emb = self.layer_preprocess(self.video_scene_flat, dim_feat=self.dim_scene)
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,1536:]
        self.video_flat = tf.concat([self.video_flat[:,:1536], self.scene_emb], -1)
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1) 
        

        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:,1:]
            loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
        tf.summary.scalar("training_loss", loss)
        
        return loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0.6):
        self.build_inputs()
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        batch_size = self.batch_size * beam_size
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [batch_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [batch_size, self.n_frames, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [batch_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        decode_length = tf.constant(decode_length)
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]    

class MV_Top_Scene_CW_Attention_Model(MV_Top_Scene_Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image,
                 dim_scene,
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(MV_Top_Scene_CW_Attention_Model, self).__init__(
                 batch_size, 
                 n_frames, 
                 dim_image,
                 dim_scene,
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit,
                 mce)
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        encoder_output["encoder_output"] = encoder_output["encoder_output"] + encoder_output["encoder_top_output"]
        # current_t = tf.shape(x)[1]
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_channelwise_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x, y)
                        # with tf.variable_scope("encdec_attention_top"):
                        #     y = common_attention.multihead_attention(
                        #         self.layer_preprocess(x_old),
                        #         encoder_output["encoder_top_output"],
                        #         encoder_decoder_pool_attention_bias,
                        #         self.dim_model,
                        #         self.dim_model,
                        #         self.dim_model,
                        #         self.num_heads,
                        #         self.attention_dropout,
                        #         attention_type="dot_product")
                        #     x_top = self.layer_postprocess(x_old, y)
                            
                        # with tf.variable_scope("layer_attention"):
                            # x_top = tf.expand_dims(x_top, 2)
                            # x = tf.expand_dims(x, 2)
                            # x = tf.concat([x_top, x], 2)
                            # x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 1, self.dim_model]), 1)
                            # x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            # y = common_attention.multihead_attention(
                            #     self.layer_preprocess(x_input),
                            #     x,
                            #     None,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.num_heads,
                            #     self.attention_dropout,
                            #     attention_type="dot_product")
                            # x = self.layer_postprocess(x_input, y)
                            # x = y
                            # x = tf.reshape(x, [-1, current_t, self.dim_model])                    
                            # y_pool5 = common_layers.conv1d(
                            #     self.layer_preprocess(x_pool5), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="SAME")
                            # x_pool5 = self.layer_postprocess(x_pool5, y_pool5)
                            # x = x + x_top
                            # x = tf.concat([x_top, x_pool5], 2)
                            # x = common_layers.conv1d(
                            #     self.layer_preprocess(x, layer_num=1), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="valid")
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)

class MV_Top_Scene_Fc7_Attention_Model(Attention_Model):
    def __init__(self,  
                 batch_size, 
                 n_frames, 
                 dim_image,
                 dim_scene,
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(MV_Top_Scene_Fc7_Attention_Model, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.layer_num = layer_num
        self.dim_scene = dim_scene
        self.mce = mce
        
        with tf.variable_scope("W_scene_encode"):
            self.encode_scene_W = tf.Variable(tf.random_normal([dim_scene, 25], 0.0, (dim_model**-0.5)), dtype=tf.float32, name="scene_w")
            self.encode_scene_b = tf.Variable(tf.zeros([25]), dtype=tf.float32, name="scene_b")
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([1536+25, dim_model], 0.0, (dim_model**-0.5)), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1024, dim_model], 0.0, (dim_model**-0.5)), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
        
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            encoder_decoder_pool_attention_bias)
        return decoder_output
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    if self.conv_before_enc == True:
                        with tf.variable_scope("conv_before"):
                            y = common_layers.conv1d(
                                self.layer_preprocess(x), 
                                self.dim_model, 
                                3, 
                                activation=tf.nn.relu, 
                                name="conv",
                                padding="SAME")
                            x = self.layer_postprocess(x, y)
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        # encoder_output["encoder_output"] = encoder_output["encoder_output"] + encoder_output["encoder_top_output"]
        # current_t = tf.shape(x)[1]
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x, y)
                            
                        with tf.variable_scope("encdec_attention_top"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_top = self.layer_postprocess(x_old, y)
                            
                        with tf.variable_scope("layer_attention"):
                            # x_top = tf.expand_dims(x_top, 2)
                            # x = tf.expand_dims(x, 2)
                            # x = tf.concat([x_top, x], 2)
                            # x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 1, self.dim_model]), 1)
                            # x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            # y = common_attention.multihead_attention(
                            #     self.layer_preprocess(x_input),
                            #     x,
                            #     None,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.dim_model,
                            #     self.num_heads,
                            #     self.attention_dropout,
                            #     attention_type="dot_product")
                            # x = self.layer_postprocess(x_input, y)
                            # x = y
                            # x = tf.reshape(x, [-1, current_t, self.dim_model])
                            
                            # y_pool5 = common_layers.conv1d(
                            #     self.layer_preprocess(x_pool5), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="SAME")
                            # x_pool5 = self.layer_postprocess(x_pool5, y_pool5)
                            x = 0.7 * x + 0.3 * x_top
                            
                            # x = tf.concat([x_top, x_pool5], 2)
                            # x = common_layers.conv1d(
                            #     self.layer_preprocess(x, layer_num=1), 
                            #     self.dim_model, 
                            #     1, 
                            #     activation=tf.nn.relu, 
                            #     name="conv",
                            #     padding="valid")
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn(self, inputs, inputs_top, targets):
        encoder_output = {}
        encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + inputs
            # encoder_top_output = encoder_top_output + inputs_top
            inputs = tf.reshape(self.layer_preprocess(inputs), [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = tf.reshape(self.layer_preprocess(inputs_top), [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias)
        return encoder_output, decoder_output
    
    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_scene = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_scene])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   
        
        
        self.video_scene_flat = tf.reshape(self.video_scene, [-1, self.dim_scene])
        self.video_scene_flat = self.layer_preprocess(self.video_scene_flat, dim_feat=self.dim_scene)
        self.scene_emb = tf.nn.xw_plus_b(self.video_scene_flat, self.encode_scene_W, self.encode_scene_b)
        self.scene_emb = self.layer_preprocess(self.scene_emb, dim_feat=25)
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,1536:]
        self.video_flat = tf.concat([self.video_flat[:,:1536], self.scene_emb], -1)
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1)
        

        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:,1:]
            loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
        tf.summary.scalar("training_loss", loss)
        
        return loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0.6):
        self.build_inputs()
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        batch_size = self.batch_size * beam_size
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [batch_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [batch_size, self.n_frames, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [batch_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        decode_length = tf.constant(decode_length)
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]    
    
    
class Hierachy_Sparse_Model(Attention_Model):
    def __init__(self, 
                 batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 layer_num,
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False,
                 mce=False):
        super(Hierachy_Sparse_Model, self).__init__(batch_size, 
                 n_frames, 
                 dim_image, 
                 n_words, 
                 vocab_size, 
                 dim_model, 
                 dim_hidden,
                 num_heads,
                 encoder_layers, 
                 decoder_layers,
                 preprocess_dropout,
                 attention_dropout,
                 bias_init_vector,
                 conv_before_enc,
                 swish_activation,
                 use_gated_linear_unit)
        
        self.layer_num = layer_num
        self.mce = mce
        
        with tf.device("/cpu:0"):
            self.word_emb = tf.Variable(tf.random_uniform([vocab_size, 300], -0.05, 0.05), name='word_emb')
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([1836, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
            
        with tf.variable_scope("W_top_encode"):
            self.encode_top_W = tf.Variable(tf.random_normal([1324, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="top_w")
            self.encode_top_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="top_b")
            
            #self.quanzhong1 = tf.Variable(tf.ones([1]), dtype=tf.float32, name="quanzhong1")
            #self.zh = tf.ones([1], dtype=tf.float32)
            #self.quan1 = tf.sigmoid(self.quanzhong1)
            
            #self.quanzhong2 = tf.Variable(tf.ones([1]), dtype=tf.float32, name="quanzhong2")
            #self.quan2 = tf.sigmoid(self.quanzhong2)
            
            
        
        if self.mce == True:
            with tf.variable_scope("W_gate_0"):
                self.W_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_0")
                self.U_gate_0 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_0")
                self.b_gate_0 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_0")

            with tf.variable_scope("W_gate_1"):
                self.W_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="W_gate_1")
                self.U_gate_1 = tf.Variable(tf.random_normal([dim_model, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="U_gate_1")
                self.b_gate_1 = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="b_gate_1")

    def layer_preprocess(self, layer_input, layer_num=0, dim_feat=None):
        if dim_feat:
            return common_layers.layer_prepostprocess(
                None,
                layer_input,
                sequence="n",
                dropout_rate=self.preprocess_dropout,
                norm_type="layer",
                depth=dim_feat,
                epsilon=1e-6,
                default_name="layer_preprocess")
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model * (layer_num + 1),
            epsilon=1e-6,
            default_name="layer_preprocess")
    
    def encode(self, inputs, num_encode_layers=0, name="encoder"):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output, encoder_decoder_attention_bias
    
    def encode2(self, input_image, input_motion, num_encode_layers=0, name="encoder"):
        encoder_input_image, self_attention_bias_image, encoder_decoder_attention_bias_image = (
            self.model_prepare_encoder(input_image))
        
        encoder_input_motion, self_attention_bias_motion, encoder_decoder_attention_bias_motion = (
            self.model_prepare_encoder(input_motion))
        
        
        
        encoder_output_image, encoder_output_motion = self.model_encoder_boundary(
            encoder_input_image,
            encoder_input_motion,
            self_attention_bias_image,
            self_attention_bias_motion,
            num_encode_layers=num_encode_layers,
            name=name)
        return encoder_output_image, encoder_output_motion, encoder_decoder_attention_bias_image, encoder_decoder_attention_bias_motion
    
    
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, encoder_decoder_pool_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder_boundary(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            encoder_decoder_pool_attention_bias)
        return decoder_output
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        x = encoder_input
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        
                        #y = SparseSelfBoundaryAttention_(self.layer_preprocess(x), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            encoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    
    def model_encoder_cross(self,
                      encoder_image,
                      encoder_motion,
                      encoder_image_bias,
                      encoder_motion_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        #x = encoder_input
        x_image = encoder_image
        x_motion = encoder_motion
        
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    
                    with tf.variable_scope("self_attention_image"):
                        y_image = SparseSelfAttention_(self.layer_preprocess(x_image), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        x_image = self.layer_postprocess(x_image, y_image)
                    with tf.variable_scope("ffn_image"):
                        y_image = self.model_ffn_layer(self.layer_preprocess(x_image))
                        x_image = self.layer_postprocess(x_image, y_image)
                        #x_image = self.layer_preprocess(x_image)
                        
                    with tf.variable_scope("self_attention_motion"):
                        y_motion = SparseSelfAttention_(self.layer_preprocess(x_motion),self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        x_motion = self.layer_postprocess(x_motion, y_motion)
                    with tf.variable_scope("ffn_motion"):
                        y_motion = self.model_ffn_layer(self.layer_preprocess(x_motion))
                        x_motion = self.layer_postprocess(x_motion, y_motion)
                        #x_motion = self.layer_preprocess(x_motion)
                        
                    with tf.variable_scope("cross_attention_image_motion"):
                        y_image = SparseCrossAttention_(self.layer_preprocess(x_image), self.layer_preprocess(x_motion), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        cross_image = self.layer_postprocess(x_image, y_image)
                        
                    with tf.variable_scope("cross_attention_motion_image"):
                        y_motion = SparseCrossAttention_(self.layer_preprocess(x_motion), self.layer_preprocess(x_image), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        cross_motion = self.layer_postprocess(x_motion, y_motion)
                    
                    with tf.variable_scope("cross_ffn"):
                        y_image = self.model_ffn_layer(self.layer_preprocess(cross_image))
                        y_motion = self.model_ffn_layer(self.layer_preprocess(cross_motion))
                        
                        x_image = self.layer_postprocess(cross_image, y_image)
                        x_motion = self.layer_postprocess(cross_motion, y_motion)
                    
        return self.layer_preprocess(x_image), self.layer_preprocess(x_motion)
    
    def model_encoder_maxpool(self,
                      encoder_image,
                      encoder_motion,
                      encoder_image_bias,
                      encoder_motion_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        #x = encoder_input
        x_image = encoder_image
        x_motion = encoder_motion
        
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    
                    with tf.variable_scope("self_attention_image"):
                        y_image = SparseSelfMaxPoolAttention_(self.layer_preprocess(x_image), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        x_image = self.layer_postprocess(x_image, y_image)
                    with tf.variable_scope("ffn_image"):
                        y_image = self.model_ffn_layer(self.layer_preprocess(x_image))
                        x_image = self.layer_postprocess(x_image, y_image)
                        #x_image = self.layer_preprocess(x_image)
                        
                    with tf.variable_scope("self_attention_motion"):
                        y_motion = SparseSelfMaxPoolAttention_(self.layer_preprocess(x_motion),self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        x_motion = self.layer_postprocess(x_motion, y_motion)
                    with tf.variable_scope("ffn_motion"):
                        y_motion = self.model_ffn_layer(self.layer_preprocess(x_motion))
                        x_motion = self.layer_postprocess(x_motion, y_motion)
                        #x_motion = self.layer_preprocess(x_motion)
                        
                    with tf.variable_scope("cross_attention_image_motion"):
                        y_image = SparseMaxPoolAttention_(self.layer_preprocess(x_image), self.layer_preprocess(x_motion), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        cross_image = self.layer_postprocess(x_image, y_image)
                        
                    with tf.variable_scope("cross_attention_motion_image"):
                        y_motion = SparseMaxPoolAttention_(self.layer_preprocess(x_motion), self.layer_preprocess(x_image), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        cross_motion = self.layer_postprocess(x_motion, y_motion)
                    
                    with tf.variable_scope("cross_ffn"):
                        y_image = self.model_ffn_layer(self.layer_preprocess(cross_image))
                        y_motion = self.model_ffn_layer(self.layer_preprocess(cross_motion))
                        
                        x_image = self.layer_postprocess(cross_image, y_image)
                        x_motion = self.layer_postprocess(cross_motion, y_motion)
                    
        return self.layer_preprocess(x_image), self.layer_preprocess(x_motion)
    
    def model_encoder_boundary(self,
                      encoder_image,
                      encoder_motion,
                      encoder_image_bias,
                      encoder_motion_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        #x = encoder_input
        x_image = encoder_image
        x_motion = encoder_motion
        
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    
                    with tf.variable_scope("self_attention_image"):
                        y_image = SparseSelfBoundaryAttention_(self.layer_preprocess(x_image), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        x_image = self.layer_postprocess(x_image, y_image)
                    with tf.variable_scope("ffn_image"):
                        y_image = self.model_ffn_layer(self.layer_preprocess(x_image))
                        x_image = self.layer_postprocess(x_image, y_image)
                        #x_image = self.layer_preprocess(x_image)
                        
                    with tf.variable_scope("self_attention_motion"):
                        y_motion = SparseSelfBoundaryAttention_(self.layer_preprocess(x_motion),self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        x_motion = self.layer_postprocess(x_motion, y_motion)
                    with tf.variable_scope("ffn_motion"):
                        y_motion = self.model_ffn_layer(self.layer_preprocess(x_motion))
                        x_motion = self.layer_postprocess(x_motion, y_motion)
                        #x_motion = self.layer_preprocess(x_motion)
                        
                    with tf.variable_scope("cross_attention_image_motion"):
                        y_image = SparseBoundaryAttention_(self.layer_preprocess(x_image), self.layer_preprocess(x_motion), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        cross_image = self.layer_postprocess(x_image, y_image)
                        
                    with tf.variable_scope("cross_attention_motion_image"):
                        y_motion = SparseBoundaryAttention_(self.layer_preprocess(x_motion), self.layer_preprocess(x_image), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        cross_motion = self.layer_postprocess(x_motion, y_motion)
                    
                    with tf.variable_scope("cross_ffn"):
                        y_image = self.model_ffn_layer(self.layer_preprocess(cross_image))
                        y_motion = self.model_ffn_layer(self.layer_preprocess(cross_motion))
                        
                        x_image = self.layer_postprocess(cross_image, y_image)
                        x_motion = self.layer_postprocess(cross_motion, y_motion)
                    
        return self.layer_preprocess(x_image), self.layer_preprocess(x_motion)
    
    def model_encoder2(self,
                      encoder_image,
                      encoder_motion,
                      encoder_image_bias,
                      encoder_motion_bias,
                      num_encode_layers=0,
                      name="encoder",
                      reuse=False):
        #x = encoder_input
        x_image = encoder_image
        x_motion = encoder_motion
        
        with tf.variable_scope(name, reuse=reuse):
            for layer in xrange(num_encode_layers):
                with tf.variable_scope("layer_%d" % layer):
                    
                    with tf.variable_scope("self_attention_image"):
                        y_image = SparseSelfGlobalAttention_(self.layer_preprocess(x_image), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        x_image = self.layer_postprocess(x_image, y_image)
                    with tf.variable_scope("ffn_image"):
                        y_image = self.model_ffn_layer(self.layer_preprocess(x_image))
                        x_image = self.layer_postprocess(x_image, y_image)
                        #x_image = self.layer_preprocess(x_image)
                        
                    with tf.variable_scope("self_attention_motion"):
                        y_motion = SparseSelfGlobalAttention_(self.layer_preprocess(x_motion),self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        x_motion = self.layer_postprocess(x_motion, y_motion)
                    with tf.variable_scope("ffn_motion"):
                        y_motion = self.model_ffn_layer(self.layer_preprocess(x_motion))
                        x_motion = self.layer_postprocess(x_motion, y_motion)
                        #x_motion = self.layer_preprocess(x_motion)
                        
                    with tf.variable_scope("cross_attention_image_motion"):
                        y_image = SparseGlobalAttention_(self.layer_preprocess(x_image), self.layer_preprocess(x_motion), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        cross_image = self.layer_postprocess(x_image, y_image)
                        
                    with tf.variable_scope("cross_attention_motion_image"):
                        y_motion = SparseGlobalAttention_(self.layer_preprocess(x_motion), self.layer_preprocess(x_image), self.num_heads, self.dim_model/self.num_heads, 4, 5)
                        cross_motion = self.layer_postprocess(x_motion, y_motion)
                    
                    with tf.variable_scope("cross_ffn"):
                        y_image = self.model_ffn_layer(self.layer_preprocess(cross_image))
                        y_motion = self.model_ffn_layer(self.layer_preprocess(cross_motion))
                        
                        x_image = self.layer_postprocess(cross_image, y_image)
                        x_motion = self.layer_postprocess(cross_motion, y_motion)
                    
        return self.layer_preprocess(x_image), self.layer_preprocess(x_motion)
    
    
        
    def model_decoder(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        current_t = tf.shape(x)[1]
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention_top"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = y
                        with tf.variable_scope("encdec_attention"):
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_top = y
                        
                        
                        with tf.variable_scope("layer_attention"):
                            x_top = tf.expand_dims(x_top, 2)
                            x = tf.expand_dims(x, 2)
                            x_cap = tf.expand_dims(x_old, 2)
                            x = tf.concat([x_top, x, x_cap], 2)
                            # x = tf.concat([x_top, x], 2)
                            x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 2, self.dim_model]), 1)
                            x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_input),
                                x,
                                None,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                1,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x_input, y)
                            x = tf.reshape(x, [-1, current_t, self.dim_model])
                        
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_decoder2(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        current_t = tf.shape(x)[1]
        
        with tf.variable_scope("cross"):
            with tf.variable_scope("m-i"):
                weights_mi, _ = ATT.multihead_attention_weights(
                                encoder_output["encoder_top_output"],
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
            with tf.variable_scope("i-m"):
                weights_im, _ = ATT.multihead_attention_weights(
                                encoder_output["encoder_output"],
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
            
        
        
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention_top"):
                            weights_image, v_image = ATT.multihead_attention_weights(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            #x = y
                        with tf.variable_scope("encdec_attention"):
                            weights_motion, v_motion = ATT.multihead_attention_weights(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            #x_top = y
                        with tf.variable_scope("combine_weights"):
                            #embed()
                            #image
                            weights_image = 1.0 * weights_image + 0.0 * tf.matmul(weights_motion, weights_mi) 
                            #motion
                            weights_motion = 1.0 * weights_motion + 0.0 * tf.matmul(weights_image, weights_im)
                            
                            image_ = tf.matmul(weights_image, v_image)
                            motion_ = tf.matmul(weights_motion, v_motion)
                            
                            image_ = tf.transpose(image_, [0, 2, 1, 3])
                            motion_ = tf.transpose(motion_, [0, 2, 1, 3])
                            
                            image_shape = common_layers.shape_list(image_)
                            motion_shape = common_layers.shape_list(motion_)
                            
                            a_i,b_i = image_shape[-2:]
                            a_m,b_m = motion_shape[-2:]
                            
                            x = tf.reshape(image_, image_shape[:-2] + [a_i * b_i])
                            x_top = tf.reshape(motion_, motion_shape[:-2] + [a_m * b_m])
                            
                            #x = tf.layers.dense(
                            #    image_, self.dim_model, use_bias=False, name="output_image")
                            
                            #x_top = tf.layers.dense(
                            #    motion_, self.dim_model, use_bias=False, name="output_motion")
                            
                        
                        
                        with tf.variable_scope("layer_attention"):
                            x_top = tf.expand_dims(x_top, 2)
                            x = tf.expand_dims(x, 2)
                            x_cap = tf.expand_dims(x_old, 2)
                            x = tf.concat([x_top, x, x_cap], 2)
                            # x = tf.concat([x_top, x], 2)
                            x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 2, self.dim_model]), 1)
                            x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_input),
                                x,
                                None,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                1,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x_input, y)
                            x = tf.reshape(x, [-1, current_t, self.dim_model])
                        
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    
    def model_decoder3(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        current_t = tf.shape(x)[1]
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention_top"):
                            
                            encoder_motion = tf.nn.relu(tf.reduce_mean(encoder_output["encoder_top_output"], 1))
                            encoder_motion = tf.sigmoid(tf.layers.dense(encoder_motion, self.dim_model, use_bias=False))
                            motion_gate = tf.expand_dims(encoder_motion, 1)
                            
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x),
                                (1 + motion_gate) * encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = y
                        with tf.variable_scope("encdec_attention"):
                            
                            encoder_image = tf.nn.relu(tf.reduce_mean(encoder_output["encoder_output"], 1))
                            encoder_image = tf.sigmoid(tf.layers.dense(encoder_image, self.dim_model, use_bias=False))
                            image_gate = tf.expand_dims(encoder_image, 1)
                            
                            
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_old),
                                (1 + image_gate) * encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x_top = y
                        
                        
                        with tf.variable_scope("layer_attention"):
                            x_top = tf.expand_dims(x_top, 2)
                            x = tf.expand_dims(x, 2)
                            x_cap = tf.expand_dims(x_old, 2)
                            x = tf.concat([x_top, x, x_cap], 2)
                            # x = tf.concat([x_top, x], 2)
                            x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 2, self.dim_model]), 1)
                            x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_input),
                                x,
                                None,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                1,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x_input, y)
                            x = tf.reshape(x, [-1, current_t, self.dim_model])
                        
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    
    
    def model_decoder_sparse(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        current_t = tf.shape(x)[1]
        image_len = tf.shape(encoder_output["encoder_output"])[1]
        motion_len = image_len
        step = 4
        
        batch_size = tf.shape(x)[0]
        
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention_top"):
                            logits_image, v_image = ATT.multihead_attention_logits(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            ## logits_image [batch_size, 8, 35, 80]                            
                            ## v_imag [batch_size, 8, 80, 64]
                            #v_image = tf.reshape(v_image, [-1, 20, 4, 1024])
                            #embed()
                            
                            logits_image = tf.reshape(logits_image, [-1, current_t, image_len/step, step])
                            index_image = tf.argmax(logits_image, axis=3)
                            max_image = tf.reduce_max(logits_image, axis=3)
                            max_image = tf.nn.softmax(max_image, 2)
                            max_image = tf.expand_dims(max_image, 3)
                            
                            index_image = tf.reshape(index_image, [-1, 1])
                            indices_image = tf.expand_dims(tf.range(0, batch_size*current_t*8*image_len/step, 1), 1)
                            #embed()
                            
                            index_image = tf.cast(index_image, tf.int32)
                            
                            concated_image = tf.concat([indices_image, index_image], 1)
                            onehot_image = tf.sparse_to_dense(concated_image, tf.stack([batch_size*current_t*8*image_len/step, step]), 1.0, 0.0)
                            onehot_image = tf.reshape(onehot_image, [-1, current_t, image_len/step, step])
                            
                            max_image = max_image * onehot_image
                            max_image = tf.reshape(max_image, [-1, current_t, image_len])
                            v_image = tf.reshape(v_image, [-1, image_len, self.dim_model/8])
                            x = tf.matmul(max_image, v_image)
                            
                            x = tf.reshape(x, [batch_size, 8, current_t, self.dim_model/8])
                            x = tf.transpose(x, [0, 2, 1, 3])
                            x = tf.reshape(x, [batch_size, current_t, self.dim_model])
                            
                        with tf.variable_scope("encdec_attention"):
                            logits_motion, v_motion = ATT.multihead_attention_logits(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            #x_top = y
                            
                            logits_motion = tf.reshape(logits_motion, [-1, current_t, motion_len/step, step])
                            index_motion = tf.argmax(logits_motion, axis=3)
                            max_motion = tf.reduce_max(logits_motion, axis=3)
                            max_motion = tf.nn.softmax(max_motion, 2)
                            max_motion = tf.expand_dims(max_motion, 3)
                            
                            index_motion = tf.reshape(index_motion, [-1, 1])
                            indices_motion = tf.expand_dims(tf.range(0, batch_size*current_t*8*motion_len/step, 1), 1)
                            #embed()
                            index_motion = tf.cast(index_motion, tf.int32)
                            
                            concated_motion = tf.concat([indices_motion, index_motion], 1)
                            onehot_motion = tf.sparse_to_dense(concated_motion, tf.stack([batch_size*current_t*8*motion_len/step, step]), 1.0, 0.0)
                            onehot_motion = tf.reshape(onehot_motion, [-1, current_t, motion_len/step, step])
                            
                            max_motion = max_motion * onehot_motion
                            max_motion = tf.reshape(max_motion, [-1, current_t, motion_len])
                            v_motion = tf.reshape(v_motion, [-1, motion_len, self.dim_model/8])
                            x_top = tf.matmul(max_motion, v_motion)
                            
                            x_top = tf.reshape(x_top, [batch_size, 8, current_t, self.dim_model/8])
                            x_top = tf.transpose(x_top, [0, 2, 1, 3])
                            x_top = tf.reshape(x_top, [batch_size, current_t, self.dim_model])
                       
                            
                        
                        
                        with tf.variable_scope("layer_attention"):
                            x_top = tf.expand_dims(x_top, 2)
                            x = tf.expand_dims(x, 2)
                            x_cap = tf.expand_dims(x_old, 2)
                            x = tf.concat([x_top, x, x_cap], 2)
                            # x = tf.concat([x_top, x], 2)
                            x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 2, self.dim_model]), 1)
                            x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_input),
                                x,
                                None,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                1,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x_input, y)
                            x = tf.reshape(x, [-1, current_t, self.dim_model])
                        
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    def model_decoder_boundary(self,
                      decoder_input,
                      encoder_output,
                      decoder_self_attention_bias,
                      encoder_decoder_attention_bias,
                      encoder_decoder_pool_attention_bias,
                      name="decoder"):
        x = decoder_input
        current_t = tf.shape(x)[1]
        image_len = tf.shape(encoder_output["encoder_output"])[1]
        motion_len = image_len
        step = 4
        
        batch_size = tf.shape(x)[0]
        
        with tf.variable_scope(name):
            for layer in xrange(self.decoder_layers):
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        y = common_attention.multihead_attention(
                            self.layer_preprocess(x),
                            None,
                            decoder_self_attention_bias,
                            self.dim_model,
                            self.dim_model,
                            self.dim_model,
                            self.num_heads,
                            self.attention_dropout,
                            attention_type="dot_product")
                        x = self.layer_postprocess(x, y)
                    x_old = x
                    if encoder_output is not None:
                        with tf.variable_scope("encdec_attention_top"):
                            logits_image, v_image = ATT.multihead_attention_logits(
                                self.layer_preprocess(x),
                                encoder_output["encoder_output"],
                                encoder_decoder_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            ## logits_image [batch_size, 8, 35, 80]                            
                            ## v_imag [batch_size, 8, 80, 64]
                            #v_image = tf.reshape(v_image, [-1, 20, 4, 1024])
                            #embed()
                            logits_image = tf.reshape(logits_image, [-1, current_t, image_len])
                            
                            sub_logits_image = logits_image[:,:,:-1]
                            sub_logits_image = tf.concat([tf.expand_dims(logits_image[:,:,0], -1), sub_logits_image], -1)
                            
                            grad_image = tf.abs(logits_image - sub_logits_image)
                            #grad_image = 5*grad_image + logits_image
                            grad_image = logits_image
                            grad_image_value, grad_image_index = tf.nn.top_k(grad_image, k=image_len//step)
                            
                            
                            grad_image_index = tf.reshape(grad_image_index, [-1, image_len//step])
                            logits_image_ = tf.reshape(logits_image, [-1, image_len])
                            grad_image_value = tf.batch_gather(logits_image_, grad_image_index)
                            
                            
                            temp_image = tf.expand_dims(v_image, 2)
                            temp_image = tf.tile(temp_image, [1, 1, current_t, 1, 1])
                            
                            temp_image = tf.reshape(temp_image, [-1, image_len, self.dim_model/8]) 
        
                            grad_image_feat = tf.batch_gather(temp_image, grad_image_index)
                            grad_image_feat = tf.reshape(grad_image_feat, [-1, 8, current_t, image_len//step, self.dim_model/8])
        
                            
                            grad_image_value = tf.nn.softmax(grad_image_value, -1)
                            grad_image_value = tf.reshape(grad_image_value, [-1, 8, current_t, 1, image_len//step])
                            #a = tf.reshape(grad_image_value, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
                            
                            x = tf.matmul(grad_image_value, grad_image_feat)
                            
                            x = tf.reshape(x, [batch_size, 8, current_t, self.dim_model/8])
                            x = tf.transpose(x, [0, 2, 1, 3])
                            x = tf.reshape(x, [batch_size, current_t, self.dim_model])
                            
                        with tf.variable_scope("encdec_attention"):
                            logits_motion, v_motion = ATT.multihead_attention_logits(
                                self.layer_preprocess(x_old),
                                encoder_output["encoder_top_output"],
                                encoder_decoder_pool_attention_bias,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                self.num_heads,
                                self.attention_dropout,
                                attention_type="dot_product")
                            #x_top = y
                            
                            logits_motion = tf.reshape(logits_motion, [-1, current_t, motion_len])
                            
                            sub_logits_motion = logits_motion[:,:,:-1]
                            sub_logits_motion = tf.concat([tf.expand_dims(logits_motion[:,:,0], -1), sub_logits_motion], -1)
                            
                            grad_motion = tf.abs(logits_motion - sub_logits_motion)
                            #grad_motion = 5*grad_motion + logits_motion
                            grad_motion = logits_motion
                            
                            grad_motion_value, grad_motion_index = tf.nn.top_k(grad_motion, k=motion_len//step)
                            
                            
                            grad_motion_index = tf.reshape(grad_motion_index, [-1, motion_len//step])
                            
                            
                            logits_motion_ = tf.reshape(logits_motion, [-1, motion_len])
                            grad_motion_value = tf.batch_gather(logits_motion_, grad_motion_index)
                            
                            temp_motion = tf.expand_dims(v_motion, 2)
                            temp_motion = tf.tile(temp_motion, [1, 1, current_t, 1, 1])
                            
                            temp_motion = tf.reshape(temp_motion, [-1, motion_len, self.dim_model/8]) 
        
                            grad_motion_feat = tf.batch_gather(temp_motion, grad_motion_index)
                            grad_motion_feat = tf.reshape(grad_motion_feat, [-1, 8, current_t, motion_len//step, self.dim_model/8])
        
                            
                            grad_motion_value = tf.nn.softmax(grad_motion_value, -1)
                            grad_motion_value = tf.reshape(grad_motion_value, [-1, 8, current_t, 1, motion_len//step])
                            #a = tf.reshape(grad_image_value, [-1, nb_head, rate, new_seq_len//rate, new_seq_len//rate])
                            
                            x_top = tf.matmul(grad_motion_value, grad_motion_feat)
                            
                            x_top = tf.reshape(x_top, [batch_size, 8, current_t, self.dim_model/8])
                            x_top = tf.transpose(x_top, [0, 2, 1, 3])
                            x_top = tf.reshape(x_top, [batch_size, current_t, self.dim_model])
                       
                            
                        
                        
                        with tf.variable_scope("layer_attention"):
                            x_top = tf.expand_dims(x_top, 2)
                            x = tf.expand_dims(x, 2)
                            x_cap = tf.expand_dims(x_old, 2)
                            x = tf.concat([x_top, x, x_cap], 2)
                            # x = tf.concat([x_top, x], 2)
                            x = tf.squeeze(tf.reshape(x, [-1, 1, self.layer_num + 2, self.dim_model]), 1)
                            x_input = tf.reshape(x_old, [-1, 1, self.dim_model])
                            y = common_attention.multihead_attention(
                                self.layer_preprocess(x_input),
                                x,
                                None,
                                self.dim_model,
                                self.dim_model,
                                self.dim_model,
                                1,
                                self.attention_dropout,
                                attention_type="dot_product")
                            x = self.layer_postprocess(x_input, y)
                            x = tf.reshape(x, [-1, current_t, self.dim_model])
                        
                    with tf.variable_scope("ffn"):
                        y = self.model_ffn_layer(self.layer_preprocess(x))
                        x = self.layer_postprocess(x, y)
        return self.layer_preprocess(x)
    
    
    
    
    def model_ffn_layer(self, x):
        if self.swish_activation == True:
            conv_output = common_layers.conv_hidden_swish(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        if self.use_gated_linear_unit == True:
            conv_output = common_layers.conv_hidden_glu(
                x,
                self.dim_hidden,
                self.dim_model,
                dropout=self.attention_dropout)
            return conv_output
        conv_output = common_layers.conv_hidden_relu(
            x,
            self.dim_hidden,
            self.dim_model,
            dropout=self.attention_dropout)
        return conv_output
    
    def model_fn2(self, inputs, inputs_top, targets):
        encoder_output = {}
        encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + self.layer_preprocess(inputs)
            # encoder_top_output = encoder_top_output + self.layer_preprocess(inputs_top)
            inputs = self.layer_preprocess(inputs)
            inputs = tf.reshape(inputs, [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = self.layer_preprocess(inputs_top)
            inputs_top = tf.reshape(inputs_top, [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias)
        return encoder_output, decoder_output
    
    def model_fn(self, inputs, inputs_top, targets):
        encoder_output = {}
        #encoder_image_output, encoder_decoder_attention_bias = self.encode(inputs, num_encode_layers=self.encoder_layers)
        #encoder_top_output, encoder_decoder_top_attention_bias = self.encode(inputs_top, num_encode_layers=self.encoder_layers, name="encode_top")
        
        encoder_image_output, encoder_top_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias = self.encode2(inputs, inputs_top, num_encode_layers=self.encoder_layers)
        
        if self.mce == True:
            # encoder_image_output = encoder_image_output + self.layer_preprocess(inputs)
            # encoder_top_output = encoder_top_output + self.layer_preprocess(inputs_top)
            inputs = self.layer_preprocess(inputs)
            inputs = tf.reshape(inputs, [-1, self.dim_model])
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.dim_model])
            gate_0 = tf.sigmoid(tf.matmul(inputs, self.W_gate_0) + tf.matmul(encoder_image_output, self.U_gate_0) + self.b_gate_0)
            encoder_image_output = gate_0 * inputs + (1 - gate_0) * encoder_image_output
            encoder_image_output = tf.reshape(encoder_image_output, [-1, self.n_frames, self.dim_model])
            
            inputs_top = self.layer_preprocess(inputs_top)
            inputs_top = tf.reshape(inputs_top, [-1, self.dim_model])
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.dim_model])
            gate_1 = tf.sigmoid(tf.matmul(inputs_top, self.W_gate_1) + tf.matmul(encoder_top_output, self.U_gate_1) + self.b_gate_1)
            encoder_top_output = gate_1 * inputs_top + (1 - gate_1) * encoder_top_output
            encoder_top_output = tf.reshape(encoder_top_output, [-1, self.n_frames, self.dim_model])
        
        encoder_output["encoder_output"] = encoder_image_output
        encoder_output["encoder_top_output"] = encoder_top_output
        
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias, encoder_decoder_top_attention_bias)
        return encoder_output, decoder_output
    
    
    def weight_decay(self, decay_rate=1e-4):
        vars_list = tf.trainable_variables()
        weight_decays = []
        for v in vars_list:
            is_bias = len(v.shape.as_list()) == 1 or v.name.endswith("bias:0")
            is_emb = v.name.endswith("emb:0")
            if not (is_bias or is_emb):
                v_loss = tf.nn.l2_loss(v)
                weight_decays.append(v_loss)
        return tf.add_n(weight_decays) * decay_rate
    
    def build_inputs(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])   
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.video_flat = self.layer_preprocess(self.video_flat, dim_feat=self.dim_image)
        self.feat_top = self.video_flat[:,1836:]
        self.video_flat = self.video_flat[:,:1836]
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1) 
        # self.image_emb = tf.nn.dropout(self.image_emb, keep_prob=0.5)

        with tf.variable_scope("inputs_top"):
            self.top_emb = tf.nn.xw_plus_b(self.feat_top, self.encode_top_W, self.encode_top_b)
            self.top_emb = tf.reshape(self.top_emb, [self.batch_size, self.n_frames, self.dim_model])
            self.top_emb = self.top_emb * tf.expand_dims(self.video_mask, -1) 
            # self.top_emb = tf.nn.dropout(self.top_emb, keep_prob=0.5)
        
    def build_train_model(self):
        self.build_inputs()
        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        
        #print self.sent_emb.shape
        
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        # self.sent_emb = tf.nn.dropout(self.sent_emb, keep_prob=0.5)
        
        #print self.sent_emb
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.top_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        #print self.logits
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        #print self.labels
        
        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            #print loss
            
            loss = loss * self.caption_mask[:,1:]
            
            #print loss
            # loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
            loss = tf.reduce_sum(loss) / self.batch_size
            
            #print loss
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
            #print reg_loss
        total_loss = loss + reg_loss
        #print total_loss
        
        #embed()
        
        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("total_loss", total_loss)
        return loss
    
    def greedy_decode(self, decode_length):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(self.image_emb, self.top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size, -1, self.vocab_size])
            return logits
        
        def inner_loop(i, decoded_ids, logits):
            logits = symbols_to_logits_fn(decoded_ids)
            next_id = tf.expand_dims(tf.argmax(logits[:,-1], axis=-1), axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i+1, decoded_ids, logits
        
        decoded_ids = tf.zeros([self.batch_size, 1], dtype=tf.int64)
        _, decoded_ids, logits = tf.while_loop(
            lambda i, *_: tf.less(i, decode_length),
            inner_loop,
            [tf.constant(0), decoded_ids, tf.zeros([self.batch_size,1,1])],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None])
            ])
        
        return decoded_ids, logits

    def beam_search_decode(self, decode_length, beam_size=4, alpha=0):
        self.build_inputs()
        # decode_length = tf.shape(inputs)[1] + decode_length
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        top_emb = self.top_emb
        top_emb = tf.expand_dims(top_emb, 1)
        top_emb = tf.tile(top_emb, [1, beam_size, 1, 1])
        top_emb = tf.reshape(top_emb, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, top_emb, targets)
            decode_output = tf.reshape(decode_output, [-1, self.dim_model])
            logits = tf.nn.xw_plus_b(decode_output, self.softmax_W, self.softmax_b)
            logits = tf.reshape(logits, [self.batch_size * beam_size, -1, self.vocab_size])
            
            current_output_position = tf.shape(ids)[1] - 1
            logits = logits[:, current_output_position, :]
            return logits
        
        initial_ids = tf.zeros([self.batch_size], dtype=tf.int32)
        decode_length = tf.constant(decode_length)
        
        decoded_ids, _ = beam_search.beam_search(symbols_to_logits_fn,
                                                 initial_ids,
                                                 beam_size,
                                                 decode_length,
                                                 self.vocab_size,
                                                 alpha)
        
        return decoded_ids[:, 0]    
    
   