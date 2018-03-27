import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy


def residual_block(input_tensor,
                   is_training,
                   batch_norm=False,
                   name="",
                   reuse=False):
    """
    @brief      Create a residual block and apply it to the input tensor

    @param      input_tensor  The input tensor
    @param      kernel        Size of convolutional kernel to apply
    @param      n_filters     Number of output filters

    @return     { Tensor with the residual network applied }
    """

    # Residual block has the identity path summed with the output of
    # BN/Relu/Conv2d applied twice
    with tf.variable_scope(name):
        x = input_tensor

        # Assuming channels last here:
        n_filters = x.shape[-1]

        with tf.variable_scope(name + "_0"):
            # Batch normalization is applied first:
            if batch_norm:
                x = tf.layers.batch_normalization(x,
                                                  training=is_training,
                                                  trainable=is_training,
                                                  name="BatchNorm",
                                                  reuse=reuse)

            # Conv2d:
            x = tf.layers.conv2d(x, n_filters,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 trainable=is_training,
                                 name="Conv2D",
                                 reuse=reuse)

            # ReLU:
            x = tf.nn.relu(x)

        # Apply everything a second time:
        with tf.variable_scope(name + "_1"):

            # Batch normalization is applied first:
            if batch_norm:
                x = tf.layers.batch_normalization(x,
                                                  training=is_training,
                                                  trainable=is_training,
                                                  name="BatchNorm",
                                                  reuse=reuse)


            # Conv2d:
            x = tf.layers.conv2d(x,
                                 n_filters,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 trainable=is_training,
                                 name="Conv2D",
                                 reuse=reuse)

            # ReLU:
            x = tf.nn.relu(x)

        # Sum the input and the output:
        with tf.variable_scope(name+"_add"):
            x = tf.add(x, input_tensor, name="Add")

    return x


def downsample_block(input_tensor,
                     is_training,
                     batch_norm=False,
                     name="",
                     reuse = False):
    """
    @brief      Create a residual block and apply it to the input tensor

    @param      input_tensor  The input tensor
    @param      kernel        Size of convolutional kernel to apply
    @param      n_filters     Number of output filters

    @return     { Tensor with the residual network applied }
    """

    # Residual block has the identity path summed with the output of
    # BN/Relu/Conv2d applied twice
    with tf.variable_scope(name):

        x = input_tensor

        # Assuming channels last here:
        n_filters = 2*x.get_shape().as_list()[-1]

        with tf.variable_scope(name + "_0"):
            # Batch normalization is applied first:
            if batch_norm:
                x = tf.layers.batch_normalization(x,
                                                  training=is_training,
                                                  trainable=is_training,
                                                  name="BatchNorm",
                                                  reuse=reuse)


            # Conv2d:
            x = tf.layers.conv2d(x, n_filters,
                                 kernel_size=[3, 3],
                                 strides=[2, 2],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 trainable=is_training,
                                 name="Conv2D",
                                 reuse=reuse)

            # ReLU:
            x = tf.nn.relu(x)

        # Apply everything a second time:
        with tf.variable_scope(name + "_1"):

            # Batch normalization is applied first:
            if batch_norm:
                x = tf.layers.batch_normalization(x,
                                                  training=is_training,
                                                  trainable=is_training,
                                                  name="BatchNorm",
                                                  reuse=reuse)

            # Conv2d:
            x = tf.layers.conv2d(x,
                                 n_filters,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=None,  # automatically uses Xavier initializer
                                 kernel_regularizer=None,
                                 activity_regularizer=None,
                                 trainable=is_training,
                                 name="Conv2D",
                                 reuse=reuse)

            # ReLU:
            x = tf.nn.relu(x)

        # # Map the input tensor to the output tensor with a 1x1 convolution
        # with tf.variable_scope(name+"identity"):
        #     y = tf.layers.conv2d(input_tensor,
        #                          n_filters,
        #                          kernel_size=[1, 1],
        #                          strides=[2, 2],
        #                          padding='same',
        #                          activation=None,
        #                          use_bias=False,
        #                          trainable=is_training,
        #                          name="Conv2D1x1",
        #                          reuse=reuse)

        # # Sum the input and the output:
        # with tf.variable_scope(name+"_add"):
        #     x = tf.add(x, y)

    return x



def upsample_block(input_tensor,
                   is_training,
                   batch_norm=False,
                   n_output_filters=0,
                   name="",
                   reuse=False):
    """
    @brief      Create a residual block and apply it to the input tensor

    @param      input_tensor  The input tensor
    @param      kernel        Size of convolutional kernel to apply
    @param      n_filters     Number of output filters

    @return     { Tensor with the residual network applied }
    """

    # Residual block has the identity path summed with the output of
    # BN/Relu/Conv2d applied twice
    with tf.variable_scope(name):

        x = input_tensor

        # Assuming channels last here:
        n_filters = int(0.5*input_tensor.get_shape().as_list()[-1])
        if n_filters == 0:
          n_filters = 1

        if n_output_filters == 0:
          n_output_filters = n_filters

        with tf.variable_scope(name + "_0"):
            # Batch normalization is applied first:
            if batch_norm:
                x = tf.layers.batch_normalization(x,
                                                  training=is_training,
                                                  trainable=is_training,
                                                  name="BatchNorm",
                                                  reuse=reuse)

            # Conv2d:
            x = tf.layers.conv2d_transpose(x, n_output_filters,
                                 kernel_size=[5,5],
                                 strides=[2, 2],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 trainable=is_training,
                                 name="Conv2DTrans",
                                 reuse=None)

            # ReLU:
            x = tf.nn.relu(x)

        # Apply everything a second time:
        with tf.variable_scope(name + "_1"):

             # Batch normalization is applied first:
            if batch_norm:
                x = tf.layers.batch_normalization(x,
                                                  training=is_training,
                                                  trainable=is_training,
                                                  name="BatchNorm",
                                                  reuse=reuse)


            # Conv2d:
            x = tf.layers.conv2d(x,
                                 n_output_filters,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 trainable=is_training,
                                 name="Conv2D",
                                 reuse=None)

            # ReLU:
            x = tf.nn.relu(x)

        # # Map the input tensor to the output tensor with a 1x1 convolution
        # with tf.variable_scope(name+"identity"):
        #     y = tf.layers.conv2d_transpose(input_tensor,
        #                          n_output_filters,
        #                          kernel_size=[1, 1],
        #                          strides=[2, 2],
        #                          padding='same',
        #                          activation=None,
        #                          use_bias=False,
        #                          trainable=is_training,
        #                          name="Conv2DTrans1x1",
        #                          reuse=None)

        # # Sum the input and the output:
        # with tf.variable_scope(name+"_add"):
        #     x = tf.add(x, y)

    return x