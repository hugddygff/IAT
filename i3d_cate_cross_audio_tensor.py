#-*- coding: utf-8 -*-
from tokenizer.ptbtokenizer import PTBTokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import cv2
import scipy.io
from keras.preprocessing import sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cider.cider import Cider
from rouge.rouge import Rouge
from bleu.bleu import Bleu
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.python.layers import core as layers_core
from process_word import load_bin_vec, add_unknown_words, get_W
#import modules
from IPython import embed

import attention_model
import multi_view_attention_model_audio_cross_super_att

tf.set_random_seed(20190926)

    
#=====================================================================================
# Global Parameters
#=====================================================================================
video_path = '/data2/jintao/YouTubeClips'

video_train_feat_path = '/data1/jintao/msrvtt_merge/temporal_train'
video_test_feat_path = '/data1/jintao/msrvtt_merge/temporal_test'
video_val_feat_path = '/data1/jintao/msrvtt_merge/temporal_val'

video_train_3d_path = '/data1/jintao/msrvtt_merge/i3d_train'
video_test_3d_path = '/data1/jintao/msrvtt_merge/i3d_test'
video_val_3d_path = '/data1/jintao/msrvtt_merge/i3d_val'

video_train_mfcc_path = '/data1/jintao/msrvtt_merge/wav_train'
video_test_mfcc_path = '/data1/jintao/msrvtt_merge/wav_test'
video_val_mfcc_path = '/data1/jintao/msrvtt_merge/wav_val'

video_train_cate_path = '/data1/jintao/msrvtt_merge/category_train'
video_test_cate_path = '/data1/jintao/msrvtt_merge/category_test'
video_val_cate_path = '/data1/jintao/msrvtt_merge/category_val'

image_pred_train_path = '/data1/jintao/msrvtt_merge/image_pred_train'
image_pred_val_path = '/data1/jintao/msrvtt_merge/image_pred_val'
image_pred_test_path = '/data1/jintao/msrvtt_merge/image_pred_test'

i3d_pred_train_path = '/data1/jintao/msrvtt_merge/i3d_pred_train'
i3d_pred_val_path = '/data1/jintao/msrvtt_merge/i3d_pred_val'
i3d_pred_test_path = '/data1/jintao/msrvtt_merge/i3d_pred_test'

video_train_data_path = '/data1/jintao/msrvtt_global_local/coco_eval/msrvtt_data.csv'
video_test_data_path = '/data1/jintao/msrvtt_global_local/coco_eval/msrvtt_data.csv'
video_val_data_path = '/data1/jintao/msrvtt_global_local/coco_eval/msrvtt_data.csv'

model_path = '/data1/jintao/msrvtt_global_local/train_model/i3d_cate_audio_tensor'


if not os.path.exists(model_path):
    os.mkdir(model_path)

#=======================================================================================
# Train Parameters
#=======================================================================================


use_i3d = True
dim_image = 2560
dim_model = 512
dim_hidden = 2048
n_frames = 80
n_audio = 35
dim_audio = 128
n_words = 35
num_heads = 8
encoder_layers = 4
decoder_layers = 4
preprocess_dropout = 0.3
attention_dropout = 0.3
layer_num = 1
conv_before_enc = False
swish_activation = False
use_gated_linear_unit = False

# n_epochs = 300
n_epochs = 81
batch_size = 32
learning_rate = 1e-4
beam_size = 5
alpha = 0.8




def get_video_train_data(video_data_path, video_feat_path, video_3d_path, video_mfcc_path, video_cate_path, image_pred_path, i3d_pred_path):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data['video_path'] = video_data.apply(lambda row: row['video_id'] + '.mp4.npy', axis=1)
    video_data['video_path_mean'] = video_data.apply(lambda row: row['video_id'] + '.mp4mean.npy', axis=1)
    
    video_data['ori'] = video_data.apply(lambda row: row['video_id'], axis=1)
    
    video_data['video_3d'] = video_data['video_path'].map(lambda x: os.path.join(video_3d_path, x))   
    video_data['video_mfcc'] = video_data['video_path'].map(lambda x: os.path.join(video_mfcc_path, x))
    video_data['video_cate'] = video_data['video_path'].map(lambda x: os.path.join(video_cate_path, x))
    
    video_data['image_pred'] = video_data['video_path_mean'].map(lambda x: os.path.join(image_pred_path, x))
    video_data['i3d_pred'] = video_data['video_path_mean'].map(lambda x: os.path.join(i3d_pred_path, x))
    
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['caption'].map(lambda x: isinstance(x, str))]
    
    unique_filenames = sorted(video_data['video_path'].unique())
    train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    return train_data

def get_video_test_data(video_data_path, video_feat_path, video_3d_path, video_mfcc_path, video_cate_path, image_pred_path, i3d_pred_path):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data['video_path'] = video_data.apply(lambda row: row['video_id'] + '.mp4.npy', axis=1)
    video_data['video_path_mean'] = video_data.apply(lambda row: row['video_id'] + '.mp4mean.npy', axis=1)
    
    video_data['ori'] = video_data.apply(lambda row: row['video_id'], axis=1)
    
    video_data['video_3d'] = video_data['video_path'].map(lambda x: os.path.join(video_3d_path, x))
    video_data['video_mfcc'] = video_data['video_path'].map(lambda x: os.path.join(video_mfcc_path, x))
    video_data['video_cate'] = video_data['video_path'].map(lambda x: os.path.join(video_cate_path, x))
    
    
    video_data['image_pred'] = video_data['video_path_mean'].map(lambda x: os.path.join(image_pred_path, x))
    video_data['i3d_pred'] = video_data['video_path_mean'].map(lambda x: os.path.join(i3d_pred_path, x))
    
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['caption'].map(lambda x: isinstance(x, str))]
    
    unique_filenames = sorted(video_data['video_path'].unique())
    test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    return test_data

def get_video_val_data(video_data_path, video_feat_path, video_3d_path, video_mfcc_path, video_cate_path, image_pred_path, i3d_pred_path):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data['video_path'] = video_data.apply(lambda row: row['video_id'] + '.mp4.npy', axis=1)
    video_data['video_path_mean'] = video_data.apply(lambda row: row['video_id'] + '.mp4mean.npy', axis=1)
    
    video_data['ori'] = video_data.apply(lambda row: row['video_id'], axis=1)
    
    video_data['video_3d'] = video_data['video_path'].map(lambda x: os.path.join(video_3d_path, x))
    video_data['video_mfcc'] = video_data['video_path'].map(lambda x: os.path.join(video_mfcc_path, x))
    video_data['video_cate'] = video_data['video_path'].map(lambda x: os.path.join(video_cate_path, x))
    
    video_data['image_pred'] = video_data['video_path_mean'].map(lambda x: os.path.join(image_pred_path, x))
    video_data['i3d_pred'] = video_data['video_path_mean'].map(lambda x: os.path.join(i3d_pred_path, x))
    
    
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['caption'].map(lambda x: isinstance(x, str))]
    
    unique_filenames = sorted(video_data['video_path'].unique())
    val_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    return val_data


def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold)
    word_counts = {}
    word_counts2 = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))
    
    for xxx in vocab:
        word_counts2[xxx] = word_counts[xxx]       

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<eos>'
    ixtoword[2] = '<unk>'
    ixtoword[3] = '<bos>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<eos>'] = 1
    wordtoix['<unk>'] = 2
    wordtoix['<bos>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents
    
    word_counts2['<pad>'] = nsents
    word_counts2['<bos>'] = nsents
    word_counts2['<eos>'] = nsents
    word_counts2['<unk>'] = nsents    
    

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector, word_counts2



def get_validation_bleu(sess, current_val_data, ixtoword, wordtoix, decoded_ids, model, train_data):

    loss_vall = []
    ixtoword2 = pd.Series(np.load('./data/ixtoword.npy').tolist())
    count = 0
    val_data = current_val_data
    
    batch_size = 16
    diss = {}
    a = {}
    for start, end in zip(
                range(0, len(val_data), batch_size),
                range(batch_size, len(val_data), batch_size)):

            start_time = time.time()

            current_batch = val_data[start:end]
            current_videos = current_batch['video_path'].values
            current_concate = current_batch['ori'].values
            current_i3d = current_batch['video_3d'].values
            current_mfcc = current_batch['video_mfcc'].values
            current_cate = current_batch['video_cate'].values
            
            
            current_feats = np.zeros((batch_size, n_frames, dim_image))
            current_feats_vals = map(lambda vid: np.load(vid), current_videos)
            
            current_mfcc_feats = np.zeros((batch_size, 35, 128))
            current_feats_mfcc = map(lambda vid: np.load(vid), current_mfcc)
            
            current_cate_feats = np.zeros((batch_size, 80, 300))
            current_feats_cate = map(lambda vid: np.load(vid), current_cate)
            
            
            

            current_video_masks = np.zeros((batch_size, n_frames))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1
                
            for ind,feat in enumerate(current_feats_mfcc):
                current_mfcc_feats[ind][:len(current_feats_mfcc[ind])] = feat
                
            for ind,feat in enumerate(current_feats_cate):
                current_cate_feats[ind][:len(current_feats_cate[ind])] = feat
                
                
            
            current_cate_feats2 = current_cate_feats[:,:35,:]
            
            
            current_feats = np.concatenate([current_cate_feats, current_feats, current_cate_feats], 2)
            current_audio_feats = np.concatenate([current_mfcc_feats, current_cate_feats2], 2)
            
            
            
            generated_word_index = sess.run(decoded_ids, feed_dict={model.video: current_feats,
                                                 model.audio: current_audio_feats,
                                                 model.video_mask: current_video_masks})
            
            
            for i in range(batch_size):
                currentt_videos = train_data[train_data['video_path'] == current_videos[i]]
                
                
                currentt_captions = currentt_videos['caption'].values
                #from IPython import embed; embed()
                currentt_captions = map(lambda x: x.replace('.', ''), currentt_captions)
                currentt_captions = map(lambda x: x.replace(',', ''), currentt_captions)
                currentt_captions = map(lambda x: x.replace('"', ''), currentt_captions)
                currentt_captions = map(lambda x: x.replace('\n', ''), currentt_captions)
                currentt_captions = map(lambda x: x.replace('?', ''), currentt_captions)
                currentt_captions = map(lambda x: x.replace('!', ''), currentt_captions)
                currentt_captions = map(lambda x: x.replace('\\', ''), currentt_captions)
                currentt_captions = map(lambda x: x.replace('/', ''), currentt_captions)
                currentt_captions = map(lambda x: x.replace('  ', ' '), currentt_captions)
                
                #currentt_captions = map(lambda x: x.lower(), currentt_captions)
                for idx, each_cap in enumerate(currentt_captions):
                    word = each_cap.lower()
                    if word[0] == '':
                        word = word[1:]
                    
                    
                    currentt_captions[idx] = word            #dui zhengge danci caozuo
                    
                
                diss[count] = list(currentt_captions)
                
                gen = ixtoword2[generated_word_index[i]]
                punctuation = np.argmax(np.array(gen) == '<eos>') + 1
                gen = gen[:punctuation]

                generat_sentence = ' '.join(gen)
                generat_sentence = generat_sentence.replace('<pad> ', '')
                generat_sentence = generat_sentence.replace(' <eos>', '')
                a[count] = [generat_sentence]
                count = count + 1
                
            
    print count
    ward, _ = Cider().compute_score(diss, a)    
    return ward



def train():
    #with tf.variable_scope(tf.get_variable_scope(),reuse=False):
    train_data = get_video_train_data(video_train_data_path, video_train_feat_path, video_train_3d_path, video_train_mfcc_path, video_train_cate_path, image_pred_train_path, i3d_pred_train_path)
    train_captions = train_data['caption'].values
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path, video_test_3d_path, video_test_mfcc_path, video_test_cate_path, image_pred_test_path, i3d_pred_test_path)
    test_captions = test_data['caption'].values
    
    val_data = get_video_val_data(video_val_data_path, video_val_feat_path, video_val_3d_path, video_val_mfcc_path, video_val_cate_path, image_pred_val_path, i3d_pred_val_path)
    val_captions = val_data['caption'].values
    
    current_epoch = tf.Variable(0)
    captions_list = list(train_captions) + list(val_captions)
    captions = np.asarray(captions_list, dtype=np.object)
    
    #embed()

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)
    captions = map(lambda x: x.replace('  ', ' '), captions)
    
    wordtoix, ixtoword, bias_init_vector, word_counts = preProBuildWordVocab(captions, word_count_threshold=3)
    
    w_file = '/data2/jintao/msrvtt_code/GoogleNews-vectors-negative300.bin'

    w2v = load_bin_vec(w_file, word_counts)
    add_unknown_words(w2v, word_counts)

    w = np.zeros((len(w2v.keys()), 300), dtype = 'float32')
    for word in w2v:
        w[wordtoix[word]] = w2v[word]
        
    print w.shape
    
    np.save("./data/wordtoix", wordtoix)
    np.save('./data/ixtoword', ixtoword)
    np.save("./data/bias_init_vector", bias_init_vector)
    
    g_train = tf.Graph()
    with g_train.as_default():
        model = multi_view_attention_model_audio_cross_super_att.Hierachy_Sparse_Model(
                batch_size=batch_size,
                n_frames=n_frames,
                dim_image=dim_image + 600,
                n_audio=n_audio,
                dim_audio=dim_audio + 300,
                n_words=n_words,
                vocab_size=len(wordtoix),
                dim_model=dim_model,
                dim_hidden=dim_hidden,
                num_heads=num_heads,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                bias_init_vector=bias_init_vector,
                preprocess_dropout=preprocess_dropout,
                attention_dropout=attention_dropout,
                layer_num = layer_num,
                conv_before_enc=conv_before_enc,
                swish_activation=swish_activation,
                use_gated_linear_unit=use_gated_linear_unit)
        
        Word_emb_holder = tf.placeholder(tf.float32, [len(wordtoix), 300])
        Word_emb_assign_op = tf.assign(model.word_emb, Word_emb_holder, name="W_emb_init")
    
        #tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_video_3d, tf_audio, tf_cate, tf_embed_placeholder, tf_embed_init = model.build_model()
        tf_loss = model.build_train_model()
        
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=g_train)
        saver = tf.train.Saver(max_to_keep=20)
    
        lr = 0.0001
        learning_rate = tf.placeholder(tf.float32, shape=[])
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss=tf_loss)
        sess.run(tf.global_variables_initializer())
        
        #saver.restore(sess,'/data1/jintao/msrvtt_global_local/train_model/i3d_cate_audio_cross1/i3d_cate_audio_cross-69')
        
        #embed()
        
    g_valid = tf.Graph()
    with g_valid.as_default():
        model_valid = multi_view_attention_model_audio_cross_super_att.Hierachy_Sparse_Model(
                #batch_size=batch_size,
                batch_size = 16,
                n_frames=n_frames,
                dim_image=dim_image + 600,
                n_audio = n_audio,
                dim_audio=dim_audio + 300,
                n_words=n_words,
                vocab_size=len(wordtoix),
                dim_model=dim_model,
                dim_hidden=dim_hidden,
                num_heads=num_heads,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                bias_init_vector=bias_init_vector,
                preprocess_dropout=preprocess_dropout,
                attention_dropout=attention_dropout,
                layer_num = layer_num,
                conv_before_enc=conv_before_enc,
                swish_activation=swish_activation,
                use_gated_linear_unit=use_gated_linear_unit)    
        
        decoded_ids, _ = model_valid.greedy_decode(decode_length=n_words)
     
        
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        
        sess_valid = tf.Session(config=config, graph=g_valid)
        
        saver_valid = tf.train.Saver()
        sess_valid.run(tf.global_variables_initializer())
        
    
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.95
    
    
    #embed()
    #sess = tf.Session(config=config)
    #sess = tf.InteractiveSession(config=config)
    
    
    
    
    sess.run(Word_emb_assign_op, feed_dict = {Word_emb_holder: w} )

    

    #saver.restore(sess,'/data1/jintao/msrvtt_base/train_model/i3d_cate_high_order_mul_new2/i3d_cate_high_order_mul_new2-76')
    
    loss_fd = open('./result_txt/i3d_cate_audio_tensor.txt', 'w')
    
    best_cider = 0
    lr_flag = 0
    total_time = 0
    
    for epoch in range(0, n_epochs):
        current_epoch = epoch
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]
        #from IPython import embed; embed()
        
        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x)) ] )
        current_train_data = current_train_data.reset_index(drop=True)
        for start, end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):

            
            current_batch = current_train_data[start:end]
            
            
            current_videos = current_batch['video_path'].values
            current_concate = current_batch['ori'].values
            current_i3d = current_batch['video_3d'].values
            current_mfcc = current_batch['video_mfcc'].values
            current_cate = current_batch['video_cate'].values
            current_image_pred = current_batch['image_pred'].values
            current_i3d_pred = current_batch['i3d_pred'].values
            

            current_feats = np.zeros((batch_size, n_frames, dim_image))
            current_feats_vals = map(lambda vid: np.load(vid), current_videos)             #buzu 80 de buling bing biaoji wei 0
            
            current_mfcc_feats = np.zeros((batch_size, 35, 128))
            current_feats_mfcc = map(lambda vid: np.load(vid), current_mfcc)
            
            current_cate_feats = np.zeros((batch_size, n_frames, 300))
            current_feats_cate = map(lambda vid: np.load(vid), current_cate)
            
            current_image_pred_feats = map(lambda vid: np.load(vid), current_image_pred)
            current_i3d_pred_feats = map(lambda vid: np.load(vid), current_i3d_pred)
            
            
            current_video_masks = np.zeros((batch_size, n_frames))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1
               
            for ind,feat in enumerate(current_feats_mfcc):
                current_mfcc_feats[ind][:len(current_feats_mfcc[ind])] = feat
                
            for ind,feat in enumerate(current_feats_cate):
                current_cate_feats[ind][:len(current_feats_cate[ind])] = feat
                
            current_cate_feats2 = current_cate_feats[:,:35,:]
            
                
            current_feats = np.concatenate([current_cate_feats, current_feats, current_cate_feats], 2)
            current_audio_feats = np.concatenate([current_mfcc_feats, current_cate_feats2], 2)  #428

            current_captions = current_batch['caption'].values
            
            current_captions = map(lambda x: '<pad> ' + x, current_captions)
            current_captions = map(lambda x: x.replace('.', ''), current_captions)
            current_captions = map(lambda x: x.replace(',', ''), current_captions)
            current_captions = map(lambda x: x.replace('"', ''), current_captions)
            current_captions = map(lambda x: x.replace('\n', ''), current_captions)
            current_captions = map(lambda x: x.replace('?', ''), current_captions)
            current_captions = map(lambda x: x.replace('!', ''), current_captions)
            current_captions = map(lambda x: x.replace('\\', ''), current_captions)
            current_captions = map(lambda x: x.replace('/', ''), current_captions)
            current_captions = map(lambda x: x.replace('  ', ' '), current_captions)
            
            
            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if word[0] == '':
                    word = word[1:]
                
                if len(word) < n_words:
                    current_captions[idx] = current_captions[idx] + ' <eos>'            #dui zhengge danci caozuo
                else:
                    new_word = ''
                    for i in range(n_words-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_words)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.array( map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) )

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1  
                
            
            start_time = time.time()
            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        model.video: current_feats,
                        model.audio: current_audio_feats,
                        model.image_pred: current_image_pred_feats,
                        model.i3d_pred: current_i3d_pred_feats,
                        model.video_mask: current_video_masks,
                        model.caption: current_caption_matrix,
                        model.caption_mask: current_caption_masks,
                        learning_rate:lr
                        })
            
            print 'idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time))
            
            total_time = total_time + time.time() - start_time
        
        saver.save(sess, os.path.join(model_path, 'i3d_cate_audio_tensor'), global_step=epoch)
        
        if epoch > 50:
        
            saver_valid.restore(sess_valid,'/data1/jintao/msrvtt_global_local/train_model/i3d_cate_audio_tensor/i3d_cate_audio_tensor-' + str(epoch))
        
            
            
            val_bleu_epoch = get_validation_bleu(sess_valid,
                        val_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))]).reset_index(drop=True),
                        ixtoword,wordtoix,
                        decoded_ids, model_valid, val_data)
        
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(val_bleu_epoch) + '\n')
                        
            print 'Epoch:  ', epoch, 'val_bleu:  ',val_bleu_epoch, ' lr: ', lr
        
        

        #if val_bleu_epoch>=0.435 :
        #    test_bleu_epoch = get_validation_bleu(sess_valid,
        #                test_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))]).reset_index(drop=True),
        #                ixtoword, wordtoix,
        #                decoded_ids, model_valid, test_data)
        #    print 'Epoch:  ', epoch, 'test_bleu:  ',test_bleu_epoch
        #    loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(test_bleu_epoch) + '\n')
            
            
        

    loss_fd.close()   
   
    

def test(model_path='/data1/jintao/msrvtt_global_local/train_model/i3d_cate_audio_tensor/i3d_cate_audio_tensor-73'):
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path, video_test_3d_path, video_test_mfcc_path, video_test_cate_path, image_pred_test_path, i3d_pred_test_path)
    test_videos = test_data['video_path'].unique()
    concate_videos = test_data['ori'].unique()
    batch_size = 13

    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())

    bias_init_vector = np.load('./data/bias_init_vector.npy')
    
    g_test = tf.Graph()
    with g_test.as_default():
        model_test = multi_view_attention_model_audio_cross_super_att.Hierachy_Sparse_Model(
                batch_size=batch_size,
                n_frames=n_frames,
                dim_image=dim_image + 600,
                n_audio=n_audio,
                dim_audio=dim_audio + 300,
                n_words=n_words,
                vocab_size=len(ixtoword),
                dim_model=dim_model,
                dim_hidden=dim_hidden,
                num_heads=num_heads,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                layer_num = layer_num,
                conv_before_enc=conv_before_enc,
                swish_activation=swish_activation,
                use_gated_linear_unit=use_gated_linear_unit)

        decoded_ids = model_test.beam_search_decode(decode_length=n_words, 
                                                    beam_size=beam_size, 
                                                    alpha=alpha)
        sess_test = tf.InteractiveSession(graph=g_test)

        saver_test = tf.train.Saver()
        saver_test.restore(sess_test, model_path)

    test_output_txt_fd = open('bi_results_tensor_73.txt', 'w')
    
    test_data = test_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
    
    for start, end in zip(
                range(0, len(test_data) + 1, batch_size),
                range(batch_size, len(test_data) + 1, batch_size)):

            start_time = time.time()

            current_batch = test_data[start:end]
            current_videos = current_batch['video_path'].values
            current_concate = current_batch['ori'].values
            current_i3d = current_batch['video_3d'].values
            current_mfcc = current_batch['video_mfcc'].values
            current_cate = current_batch['video_cate'].values
            
            
            current_feats = np.zeros((batch_size, n_frames, dim_image))
            current_feats_vals = map(lambda vid: np.load(vid), current_videos)
            
            current_mfcc_feats = np.zeros((batch_size, 35, 128))
            current_feats_mfcc = map(lambda vid: np.load(vid), current_mfcc)
            
            current_cate_feats = np.zeros((batch_size, 80, 300))
            current_feats_cate = map(lambda vid: np.load(vid), current_cate)
            

            current_video_masks = np.zeros((batch_size, n_frames))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1
                
            for ind,feat in enumerate(current_feats_mfcc):
                current_mfcc_feats[ind][:len(current_feats_mfcc[ind])] = feat
                
            for ind,feat in enumerate(current_feats_cate):
                current_cate_feats[ind][:len(current_feats_cate[ind])] = feat
                
            current_cate_feats2 = current_cate_feats[:,:35,:]
            current_audio_feats = np.concatenate([current_mfcc_feats, current_cate_feats2], 2)
            
            
            current_feats = np.concatenate([current_cate_feats, current_feats, current_cate_feats], 2)
            
            generated_word_index = sess_test.run(decoded_ids, feed_dict={model_test.video: current_feats,
                                                     model_test.audio: current_audio_feats,
                                                 model_test.video_mask: current_video_masks})
            
            #gen = ixtoword[generated_word_index[1]]
            
            #embed()
            count = 0
            
            for i in range(batch_size):
                print current_concate[count + i]
                
                generated_words = ixtoword[generated_word_index[i]]
                punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
                generated_words = generated_words[:punctuation]

                generated_sentence = ' '.join(generated_words)
                generated_sentence = generated_sentence.replace('<pad> ', '')
                generated_sentence = generated_sentence.replace(' <eos>', '')
                
                print generated_sentence,'\n'
                
                test_output_txt_fd.write(video_test_feat_path + '/'+current_concate[count+i]+'.mp4.npy' + '\n')
                test_output_txt_fd.write(generated_sentence + '\n\n')
                #embed()
                
                
            count = count + batch_size