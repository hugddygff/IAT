
# coding: utf-8

import tensorflow as tf
import numpy as np
import beam_search

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from ghm_loss import GHMC_Loss

class Attention_Model(object):
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
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False):
        
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.dim_image = dim_image
        
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.attention_dropout = attention_dropout
        self.preprocess_dropout = preprocess_dropout
        self.conv_before_enc = conv_before_enc
        self.swish_activation = swish_activation
        self.use_gated_linear_unit = use_gated_linear_unit
        
#         with tf.variable_scope("W_emb"):
#             self.word_emb = tf.Variable(tf.random_normal([vocab_size, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="word_emb")
        
        with tf.variable_scope("W_emb"):
            self.word_emb = tf.Variable(tf.random_normal([vocab_size, 300], 0.0, dim_model**-0.5), dtype=tf.float32, name="word_emb")
        
        with tf.variable_scope("W_emb_encode"):
            self.W_word = tf.Variable(tf.random_normal([300, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="word_w")
            self.b_word = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="word_b")
        
        with tf.variable_scope("W_image_encode"):
            self.encode_image_W = tf.Variable(tf.random_normal([dim_image, dim_model], 0.0, dim_model**-0.5), dtype=tf.float32, name="image_w")
            self.encode_image_b = tf.Variable(tf.zeros([dim_model]), dtype=tf.float32, name="image_b")
        
        with tf.variable_scope("W_softmax"):
            self.softmax_W = tf.Variable(tf.random_normal([dim_model, vocab_size], 0.0, dim_model**-0.5), dtype=tf.float32, name="softmax_w")
            if bias_init_vector is not None:
                self.softmax_b = tf.Variable(bias_init_vector.astype(np.float32), dtype=tf.float32, name='softmax_b')
            else:
                self.softmax_b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32, name='softmax_b')

        
    def model_prepare_encoder(self, inputs):
        encoder_input = inputs
        encoder_padding = common_attention.embedding_to_padding(encoder_input)
        ignore_padding = common_attention.attention_bias_ignore_padding(
            encoder_padding)
        encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
        return (encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias)
    
    def model_prepare_decoder(self, targets):
        decoder_input = targets
        decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
        decoder_input = common_attention.add_timing_signal_1d(decoder_input)
        return (decoder_input, decoder_self_attention_bias)
    
    def encode(self, inputs):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs))
        encoder_output = self.model_encoder(
            encoder_input, 
            self_attention_bias)
        return encoder_output, encoder_decoder_attention_bias
        
    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias)
        return decoder_output


    def decode_rec(self, decoder_input, encoder_output, encoder_decoder_attention_bias):
        decoder_input, decoder_self_attention_bias = self.model_prepare_decoder(decoder_input)
        decoder_output = self.model_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias)
        return decoder_output, decoder_self_attention_bias
    
    def layer_preprocess(self, layer_input):
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model,
            epsilon=1e-6,
            default_name="layer_preprocess")

    def layer_preprocess_with_depth(self, layer_input, depth=4032, name="layer_preprocess"):
        return common_layers.layer_prepostprocess(
            None,
            layer_input,
            sequence="n",
            dropout_rate=0.0,
            norm_type="layer",
            depth=depth,
            epsilon=1e-6,
            default_name=name)

    def layer_postprocess(self, layer_input, layer_output):
        return common_layers.layer_prepostprocess(
            layer_input,
            layer_output,
            sequence="da",
            dropout_rate=self.preprocess_dropout,
            norm_type="layer",
            depth=self.dim_model,
            epsilon=1e-6,
            default_name="layer_postprocess")
    
    def model_encoder(self,
                      encoder_input,
                      encoder_self_attention_bias,
                      name="encoder"):
        x = encoder_input
        with tf.variable_scope(name):
            for layer in xrange(self.encoder_layers):
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
    
    def model_fn(self, inputs, targets):
        encoder_output, encoder_decoder_attention_bias = self.encode(inputs)
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias)
        return encoder_output, decoder_output


    def model_fn_rec(self, inputs, targets):
        encoder_output, encoder_decoder_attention_bias = self.encode(inputs)
        decoder_output, decoder_attention_bias = self.decode_rec(targets, encoder_output, encoder_decoder_attention_bias)
        return encoder_output, decoder_output, decoder_attention_bias


    def weight_decay(self, decay_rate=5e-5):
        vars_list = tf.trainable_variables()
        weight_decays = []
        for v in vars_list:
            is_bias = len(v.shape.as_list()) == 1 or v.name.endswith("bias:0")
            is_emb = v.name.endswith("emb:0")
            if not (is_bias or is_emb):
                v_loss = tf.nn.l2_loss(v)
                weight_decays.append(v_loss)
        return tf.add_n(weight_decays) * decay_rate
    
    def build_train_model(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])

        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])
        
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b) 
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1)
        
        self.sent_emb = tf.nn.embedding_lookup(self.word_emb, self.caption[:,:-1]) # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:,:-1], -1)
        
        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.sent_emb)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])
        
        self.labels = tf.one_hot(self.caption[:,1:], self.vocab_size, axis = -1)
        
        with tf.name_scope("training_loss"):
            ghmc = GHMC_Loss(bins=3, momentum=0.5)
            loss = ghmc.calc(self.logits, self.labels, self.caption_mask)
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            # loss = loss * self.caption_mask[:,1:]
            # loss = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
        total_loss = loss + reg_loss
        
        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("total_loss", total_loss)
        return loss
    
    def triplet_loss(self, video_feats, video_mask, word_embs, sent_mask, margin=1.0):
        video_vec = tf.reduce_sum(video_feats * tf.expand_dims(video_mask, -1), axis=1) \
                    / tf.reduce_sum(video_mask, axis=1, keepdims=True)
        sent_vec = tf.reduce_sum(word_embs * tf.expand_dims(sent_mask[:, 1:], -1), axis=1) \
                    / tf.reduce_sum(sent_mask[:, 1:], axis=1, keepdims=True)
        video_vec = tf.layers.dense(video_vec, 2 * self.dim_image, activation=tf.nn.relu,
                                   kernel_initializer=tf.initializers.random_normal(stddev=self.dim_image**-0.5),
                                   name="video_hidden_layer_0")
        video_vec = tf.layers.dense(video_vec, self.dim_image, activation=None,
                                   kernel_initializer=tf.initializers.random_normal(stddev=self.dim_image**-0.5),
                                   name="video_hidden_layer_1")
        sent_vec = tf.layers.dense(sent_vec, 2 * self.dim_model, activation=tf.nn.relu,
                                   kernel_initializer=tf.initializers.random_normal(stddev=self.dim_model**-0.5),
                                   name="sent_hidden_layer_0")
        sent_vec = tf.layers.dense(sent_vec, self.dim_model, activation=None,
                                   kernel_initializer=tf.initializers.random_normal(stddev=self.dim_model**-0.5),
                                   name="sent_hidden_layer_1")
        self.sim_w = tf.get_variable("Sim_Weights",
                                     shape=[self.dim_image, self.dim_model],
                                     dtype=tf.float32,
                                     initializer=tf.initializers.random_normal(stddev=self.dim_model**-0.5))
        latent_video_vec = tf.matmul(video_vec, self.sim_w)
        latent_video_vec = tf.nn.l2_normalize(latent_video_vec, axis=1)
        sent_vec = tf.nn.l2_normalize(sent_vec, axis=1)
        sim_matrix = tf.matmul(latent_video_vec, tf.transpose(sent_vec))
        sim_positive = sim_matrix * tf.eye(self.batch_size)
        sim_negtive = sim_matrix - sim_positive - 1e5 * tf.eye(self.batch_size)
        sim_positive_row = tf.reduce_sum(sim_positive, axis=0)
        sim_negtive_row = tf.reduce_max(sim_negtive, axis=0)
        loss = tf.reduce_min(tf.maximum(tf.zeros(self.batch_size), sim_negtive_row - sim_positive_row + margin))
        return loss


    def build_discriminative_train_model(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])

        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])

        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames,
                                                     self.dim_model])  # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1)

        self.sent_emb = tf.nn.embedding_lookup(self.word_emb,
                                               self.caption[:, :-1])  # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:, :-1], -1)

        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb, self.sent_emb)
        decoder_output_flat = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(decoder_output_flat, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])

        self.labels = tf.one_hot(self.caption[:, 1:], self.vocab_size, axis=-1)

        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:, 1:]
            # loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
            loss = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
        with tf.name_scope("triplet_loss"):
            triplet_loss = self.triplet_loss(self.video, self.video_mask, self.decoder_output, self.caption_mask)
        total_loss = loss + reg_loss + 10 * triplet_loss

        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("triplet_loss", triplet_loss)
        tf.summary.scalar("total_loss", total_loss)
        return loss


    def reconstruction_global_loss(self, video_feats, video_mask, word_embs, sent_mask, decoder_self_attention_bias):
        video_vec = tf.reduce_sum(video_feats * tf.expand_dims(video_mask, -1), axis=1) \
                    / tf.reduce_sum(video_mask, axis=1, keepdims=True)
        word_embs = word_embs * tf.expand_dims(sent_mask[:, 1:], -1)
        word_context_embs = self.model_encoder(word_embs, decoder_self_attention_bias, name="encode_reconstruction")
        sent_vec = tf.reduce_sum(word_context_embs * tf.expand_dims(sent_mask[:, 1:], -1), axis=1) \
                    / tf.reduce_sum(sent_mask[:, 1:], axis=1, keepdims=True)
        sent_vec = tf.layers.dense(sent_vec, self.dim_image, activation=tf.nn.relu,
                                   kernel_initializer=tf.initializers.random_normal(stddev=self.dim_image**-0.5),
                                   name="sent_hidden_layer_0")
        sent_vec = self.layer_preprocess_with_depth(sent_vec, depth=self.dim_image, name="layer_pre_sent_vec")
        video_vec = self.layer_preprocess_with_depth(video_vec, depth=self.dim_image, name="layer_pre_video_vec")
        loss = tf.nn.l2_loss(sent_vec - video_vec, name="l2_loss")

        return loss, sent_vec, video_vec


    def build_rec_train_model(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])

        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])

        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames,
                                                     self.dim_model])  # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1)

        self.sent_emb = tf.nn.embedding_lookup(self.word_emb,
                                               self.caption[:, :-1])  # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:, :-1], -1)

        self.encoder_output, self.decoder_output, decoder_attention_bias = self.model_fn_rec(self.image_emb, self.sent_emb)
        decoder_output_flat = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(decoder_output_flat, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])

        self.labels = tf.one_hot(self.caption[:, 1:], self.vocab_size, axis=-1)

        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:, 1:]
            # loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
            loss = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
        with tf.name_scope("reconstruction_global_loss"):
            reconstruction_global_loss, sent_vec, video_vec = self.reconstruction_global_loss(self.video,
                                                                         self.video_mask,
                                                                         self.decoder_output,
                                                                         self.caption_mask,
                                                                         decoder_attention_bias)
        total_loss = loss + reg_loss + reconstruction_global_loss

        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("reconstruction_global_loss", reconstruction_global_loss)
        tf.summary.scalar("total_loss", total_loss)
        return loss, sent_vec, video_vec


    def greedy_decode(self, decode_length):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b) 
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1)
        # decode_length = tf.shape(inputs)[1] + decode_length
        
        inputs = self.image_emb
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, targets)
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
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b) 
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames, self.dim_model]) # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1)
        # decode_length = tf.shape(inputs)[1] + decode_length
        
        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])
        
        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            _, decode_output = self.model_fn(inputs, targets)
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

    def beam_search_decode_fast(self, decode_length, beam_size=4, alpha=0.6):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])
        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames,
                                                     self.dim_model])  # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1)
        # decode_length = tf.shape(inputs)[1] + decode_length

        inputs = self.image_emb
        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.tile(inputs, [1, beam_size, 1, 1])
        inputs = tf.reshape(inputs, [self.batch_size * beam_size, self.n_frames, self.dim_model])

        encoder_output, encoder_decoder_attention_bias = self.encode(inputs)

        def symbols_to_logits_fn(ids):
            targets = tf.nn.embedding_lookup(self.word_emb, ids)
            with tf.variable_scope("word_emb_linear"):
                targets = tf.reshape(targets, [-1, 300])
                targets = tf.nn.xw_plus_b(targets, self.W_word, self.b_word)
                targets = tf.reshape(targets, [self.batch_size * beam_size, -1, self.dim_model])
            decode_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias)
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

class Multitask_Attention_Model(Attention_Model):
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
                 preprocess_dropout=0.0,
                 attention_dropout=0.0,
                 bias_init_vector=None,
                 conv_before_enc=False,
                 swish_activation=False,
                 use_gated_linear_unit=False):

        super(Multitask_Attention_Model, self).__init__(batch_size,
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

        with tf.variable_scope("b_softmax"):
            self.output_bias = tf.get_variable("output_bias", shape=[dim_image], initializer=tf.zeros_initializer())

    def model_fn(self, inputs, targets, enc_dec_mask=None):
        encoder_output, encoder_decoder_attention_bias = self.encode(inputs, enc_dec_mask)
        decoder_output = self.decode(targets, encoder_output, encoder_decoder_attention_bias)
        return encoder_output, decoder_output

    def encode(self, inputs, enc_dec_mask):
        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            self.model_prepare_encoder(inputs, enc_dec_mask))
        encoder_output = self.model_encoder(
            encoder_input,
            self_attention_bias)
        return encoder_output, encoder_decoder_attention_bias

    def model_prepare_encoder(self, inputs, enc_dec_mask):
        encoder_input = inputs
        encoder_padding = common_attention.embedding_to_padding(encoder_input)
        ignore_padding = common_attention.attention_bias_ignore_padding(
            encoder_padding)
        encoder_self_attention_bias = ignore_padding
        if enc_dec_mask is not None:
            encoder_decoder_padding = 1.0 - self.video_caption_mask
            ignore_enc_dec_padding = common_attention.attention_bias_ignore_padding(
                encoder_decoder_padding)
            encoder_decoder_attention_bias = ignore_enc_dec_padding
        else:
            encoder_decoder_attention_bias = ignore_padding
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
        return (encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias)

    def build_train_model(self):
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])
        self.video_caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])

        self.masked_positions = tf.placeholder(tf.float32, [self.batch_size, self.n_frames])
        self.masked_feats = tf.placeholder(tf.float32, [self.batch_size, self.n_frames, self.dim_image])

        self.caption = tf.placeholder(tf.int64, [self.batch_size, self.n_words + 1])
        self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_words + 1])

        self.lamb = tf.placeholder(tf.float32, name="LAMBDA")

        self.video_flat = tf.reshape(self.video, [-1, self.dim_image])
        self.image_emb = tf.nn.xw_plus_b(self.video_flat, self.encode_image_W, self.encode_image_b)
        self.image_emb = tf.reshape(self.image_emb, [self.batch_size, self.n_frames,
                                                     self.dim_model])  # shape: (batch_size, n_frames, dim_model)
        self.image_emb = self.image_emb * tf.expand_dims(self.video_mask, -1)

        self.sent_emb = tf.nn.embedding_lookup(self.word_emb,
                                               self.caption[:, :-1])  # shape: (batch_size, n_words+1, dim_model)
        with tf.variable_scope("word_emb_linear"):
            self.sent_emb = tf.reshape(self.sent_emb, [-1, 300])
            self.sent_emb = tf.nn.xw_plus_b(self.sent_emb, self.W_word, self.b_word)
            self.sent_emb = tf.reshape(self.sent_emb, [self.batch_size, -1, self.dim_model])
        self.sent_emb = self.sent_emb * tf.expand_dims(self.caption_mask[:, :-1], -1)

        self.encoder_output, self.decoder_output = self.model_fn(self.image_emb,
                                                                 self.sent_emb,
                                                                 enc_dec_mask=self.video_caption_mask)
        self.decoder_output = tf.reshape(self.decoder_output, [-1, self.dim_model])
        self.logits = tf.nn.xw_plus_b(self.decoder_output, self.softmax_W, self.softmax_b)
        self.logits = tf.reshape(self.logits, [self.batch_size, self.n_words, self.vocab_size])

        self.labels = tf.one_hot(self.caption[:, 1:], self.vocab_size, axis=-1)

        with tf.variable_scope("transform"):
            output_tensor = tf.layers.dense(
                self.encoder_output,
                units=self.dim_model,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.random_normal(stddev=self.dim_model**-0.5))
            output_tensor = self.layer_preprocess(output_tensor)

        output_tensor = tf.reshape(output_tensor, shape=[-1, self.dim_model])
        denoise_logits = tf.matmul(output_tensor, self.encode_image_W, transpose_b=True)
        denoise_logits = tf.nn.bias_add(denoise_logits, self.output_bias)
        denoise_logits = tf.reshape(denoise_logits, shape=[self.batch_size, -1, self.dim_image])

        weight_loss = tf.expand_dims(self.masked_positions, -1)
        with tf.name_scope("denoise_loss"):
            denoise_loss = tf.losses.mean_squared_error(
                self.masked_feats,
                denoise_logits,
                weights=weight_loss
            )

        with tf.name_scope("training_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = loss * self.caption_mask[:, 1:]
            # loss = tf.reduce_sum(loss) / tf.reduce_sum(self.caption_mask)
            loss = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope("reg_loss"):
            reg_loss = self.weight_decay()
        total_loss = loss + self.lamb * denoise_loss

        tf.summary.scalar("training_loss", loss)
        tf.summary.scalar("denoise_loss", denoise_loss)
        # tf.summary.scalar("reg_loss", reg_loss)
        tf.summary.scalar("total_loss", total_loss)
        return total_loss



