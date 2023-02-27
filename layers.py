# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

# Dependency imports
import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

import tensorflow as tf

def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        length: scalar, length of timing signal sequence.
        channels: scalar, size of timing embeddings to create. The number of
              different timescales is equal to channels / 2.
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
          math.log(float(max_timescale) / float(min_timescale)) /
          (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
          tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    length = tf.shape(x)[1]
    channels = tf.shape(x)[4]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    signal = tf.expand_dims(tf.expand_dims(signal, 2), 3)
    return x + signal

def add_timing_signal_3d(x, min_timescale=1.0, max_timescale=1.0e4):
    length = tf.shape(x)[1]
    height = tf.shape(x)[2]
    width = tf.shape(x)[3]
    channels = tf.shape(x)[4]
    
    signal_length = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    signal_length = tf.expand_dims(tf.expand_dims(signal_length, 2), 3)
    x = x + signal_length
    
    signal_height = get_timing_signal_1d(height, channels, min_timescale, max_timescale)
    signal_height = tf.expand_dims(tf.expand_dims(signal_height, 1), 3)
    x = x + signal_height
    
    signal_width = get_timing_signal_1d(width, channels, min_timescale, max_timescale)
    signal_width = tf.expand_dims(tf.expand_dims(signal_width, 1), 2)
    x = x + signal_width
    return x