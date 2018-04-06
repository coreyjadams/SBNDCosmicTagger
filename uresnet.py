import sys
import time


import tensorflow as tf

from utils import residual_block, downsample_block, upsample_block

# Declaring exception names:
class ConfigurationException(Exception): pass
class IncompleteFeedDict(Exception): pass



# Main class
class uresnet(object):
    '''Define a network model and run training

    U resnet implementation
    '''
    def __init__(self, params):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''
        required_params =[
            'MINIBATCH_SIZE',
            'SAVE_ITERATION',
            'NUM_LABELS',
            'NPLANES',
            'N_INITIAL_FILTERS',
            'NETWORK_DEPTH',
            'RESIDUAL_BLOCKS_PER_LAYER',
            'RESIDUAL_BLOCKS_DEEPEST_LAYER',
            'BALANCE_LOSS',
            'BATCH_NORM',
            'LOGDIR',
            'BASE_LEARNING_RATE',
            'TRAINING',
            'RESTORE',
            'ITERATIONS',
            'LABEL_NAMES',
        ]

        for param in required_params:
            if param not in params:
                raise ConfigurationException("Missing paragmeter "+ str(param))

        self._params = params

    def construct_network(self, dims):
        '''Build the network model

        Initializes the tensorflow model according to the parameters
        '''

        tf.reset_default_graph()


        start = time.time()
        # Initialize the input layers:
        self._input_image  = tf.placeholder(tf.float32, dims, name="input_image")
        self._input_labels = tf.placeholder(tf.int64, dims, name="input_labels")


        labels = tf.split(self._input_labels, [1]*self._params['NPLANES'], -1, name="labels_split")
        if self._params['BALANCE_LOSS']:
            self._input_weights = tf.placeholder(tf.float32, dims, name="input_weights")
            weights = tf.split(self._input_weights, [1]*self._params['NPLANES'], -1, name="weights_split")

        # Prune off the last filter of the labels and weights:
        for p in xrange(len(labels)):
            labels[p] = tf.squeeze(labels[p], axis=-1, name="labels_squeeze_{0}".format(p))

        for p in xrange(len(weights)):
            weights[p] = tf.squeeze(weights[p], axis=-1,name="weights_squeeze_{0}".format(p))

        sys.stdout.write(" - Finished input placeholders [{0:.2}s]\n".format(time.time() - start))
        start = time.time()
        logits_by_plane = self._build_network(self._input_image)

        sys.stdout.write(" - Finished Network graph [{0:.2}s]\n".format(time.time() - start))
        start = time.time()
        # for p in xrange(len(logits_by_plane)):
        #     print "logits_by_plane[{0}].get_shape(): ".format(p) + str(logits_by_plane[p].get_shape())
        self._softmax = [tf.nn.softmax(logits) for logits in logits_by_plane ]
        self._predicted_labels = [ tf.argmax(logits, axis=-1) for logits in logits_by_plane ]

        # for p in xrange(len(self._softmax)):
        #     print "self._softmax[{0}].get_shape(): ".format(p) + str(self._softmax[p].get_shape())
        # for p in xrange(len(self._predicted_labels)):
        #     print "self._predicted_labels[{0}].get_shape(): ".format(p) + str(self._predicted_labels[p].get_shape())


        # Keep a list of trainable variables for minibatching:
        with tf.variable_scope('gradient_accumulation'):
            self._accum_vars = [tf.Variable(tv.initialized_value(),
                                trainable=False) for tv in tf.trainable_variables()]

        sys.stdout.write(" - Finished gradient accumulation [{0:.2}s]\n".format(time.time() - start))
        start = time.time()

        # Accuracy calculations:
        with tf.name_scope('accuracy'):
            self._total_accuracy   = [ [] for i in range(self._params['NPLANES']) ]
            self._non_bkg_accuracy = [ [] for i in range(self._params['NPLANES']) ]
            for p in xrange(len(self._predicted_labels)):
                self._total_accuracy[p] = tf.reduce_mean(
                    tf.cast(tf.equal(self._predicted_labels[p],
                        labels[p]), tf.float32))
                # Find the non zero labels:
                non_zero_indices = tf.not_equal(labels[p], tf.constant(0, labels[p].dtype))

                non_zero_logits = tf.boolean_mask(self._predicted_labels[p], non_zero_indices)
                non_zero_labels = tf.boolean_mask(labels[p], non_zero_indices)

                self._non_bkg_accuracy[p] = tf.reduce_mean(tf.cast(tf.equal(non_zero_logits, non_zero_labels), tf.float32))

                # Add the accuracies to the summary:
                tf.summary.scalar("Total_Accuracy_plane{0}".format(p),
                    self._total_accuracy[p])
                tf.summary.scalar("Non_Background_Accuracy_plane{0}".format(p),
                    self._non_bkg_accuracy[p])

            #Compute the total accuracy and non background accuracy for all planes:
            self._all_plane_accuracy = tf.reduce_mean(self._total_accuracy)
            self._all_plane_non_bkg_accuracy = tf.reduce_mean(self._non_bkg_accuracy)

            # Add the accuracies to the summary:
            tf.summary.scalar("All_Plane_Total_Accuracy", self._all_plane_accuracy)
            tf.summary.scalar("All_Plane_Non_Background_Accuracy", self._all_plane_non_bkg_accuracy)

        sys.stdout.write(" - Finished accuracy [{0:.2}s]\n".format(time.time() - start))
        start = time.time()

        # Loss calculations:
        with tf.name_scope('cross_entropy'):
            self._loss_by_plane = [ [] for i in range(self._params['NPLANES']) ]
            for p in xrange(len(logits_by_plane)):

                # Unreduced loss, shape [BATCH, L, W]
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[p],
                    logits=logits_by_plane[p])

                if self._params['BALANCE_LOSS']:
                    losses = tf.multiply(losses, weights)

                self._loss_by_plane[p] = tf.reduce_sum(tf.reduce_sum(losses, axis=[1,2]))

                # Add the loss to the summary:
                tf.summary.scalar("Total_Loss_plane{0}".format(p), self._loss_by_plane[p])

            self._loss = tf.reduce_sum(self._loss_by_plane)
            tf.summary.scalar("Total_Loss", self._loss)

        sys.stdout.write(" - Finished cross entropy [{0:.2}s]\n".format(time.time() - start))
        start = time.time()

        # Optimizer:
        if self._params['TRAINING']:
            with tf.name_scope("training"):
                self._global_step = tf.Variable(0, dtype=tf.int32,
                    trainable=False, name='global_step')
                if self._params['BASE_LEARNING_RATE'] <= 0:
                    opt = tf.train.AdamOptimizer()
                else:
                    opt = tf.train.AdamOptimizer(self._params['BASE_LEARNING_RATE'])

                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):

                # Variables for minibatching:
                self._zero_gradients =  [tv.assign(tf.zeros_like(tv)) for tv in self._accum_vars]
                self._accum_gradients = [self._accum_vars[i].assign_add(gv[0]) for
                                         i, gv in enumerate(opt.compute_gradients(self._loss))]
                self._apply_gradients = opt.apply_gradients(zip(self._accum_vars, tf.trainable_variables()),
                    global_step = self._global_step)

        sys.stdout.write(" - Finished optimizer [{0:.2}s]\n".format(time.time() - start))
        start = time.time()

        # Snapshotting:
        with tf.name_scope('snapshot'):
            for p in xrange(len(logits_by_plane)):

                target_img = tf.cast(tf.reshape(labels[p], labels[p].get_shape().as_list() + [1,]), tf.float32)
                target_img = tf.image.grayscale_to_rgb(target_img)
                tf.summary.image('labels_plane{0}'.format(p), target_img,max_outputs=10)

                output_img = tf.cast(tf.reshape(self._predicted_labels[p], labels[p].get_shape().as_list() + [1,]), tf.float32)
                output_img = tf.image.grayscale_to_rgb(output_img)
                tf.summary.image('logits_plane{0}'.format(p), output_img,max_outputs=10)


        # Merge the summaries:
        self._merged_summary = tf.summary.merge_all()
        sys.stdout.write(" - Finished snapshotting [{0:.2}s]\n".format(time.time() - start))


    def apply_gradients(self,sess):

        return sess.run( [self._apply_gradients], feed_dict = {})


    def feed_dict(self, images, labels, weights=None):
        '''Build the feed dict

        Take input images, labels and (optionally) weights and match
        to the correct feed dict tensorrs

        Arguments:
            images {numpy.ndarray} -- Image array, [BATCH, L, W, F]
            labels {numpy.ndarray} -- Label array, [BATCH, L, W, F]

        Keyword Arguments:
            weights {numpy.ndarray} -- (Optional) input weights, same shape as labels (default: {None})

        Returns:
            [dict] -- Feed dictionary for a tf session run call

        Raises:
            IncompleteFeedDict -- If weights are requested in the configuration but not provided.
        '''
        fd = dict()
        fd.update({self._input_image : images})
        if labels is not None:
            fd.update({self._input_labels : labels})
        if self._params['TRAINING'] and self._params['BALANCE_LOSS']:
            if weights is None:
                raise IncompleteFeedDict("Missing Weights when loss balancing requested.")
            fd.update({self._input_weights : weights})

        return fd

    def losses():
        pass

    def make_summary(self, sess, input_data, input_label, input_weight=None):
        fd = self.feed_dict(images  = input_data,
                            labels  = input_label,
                            weights = input_weight)
        return sess.run(self._merged_summary, feed_dict=fd)

    def zero_gradients(self, sess):
        sess.run(self._zero_gradients)

    def accum_gradients(self, sess, input_data, input_label, input_weight=None):

        feed_dict = self.feed_dict(images  = input_data,
                                   labels  = input_label,
                                   weights = input_weight)

        ops = [self._accum_gradients]
        doc = ['']
        # classification
        ops += [self._loss, self._all_plane_accuracy, self._all_plane_non_bkg_accuracy]
        doc += ['loss', 'acc. all', 'acc. nonzero']

        return sess.run(ops, feed_dict = feed_dict ), doc


    def run_test(self,sess, input_data, input_label, input_weight=None):
        feed_dict = self.feed_dict(images   = input_data,
                                   labels   = input_label,
                                   weights  = input_weight)

        ops = [self._loss, self._all_plane_accuracy, self._all_plane_non_bkg_accuracy]
        doc = ['loss', 'acc. all', 'acc. nonzero']

        return sess.run(ops, feed_dict = feed_dict ), doc

    def inference(self,sess,input_data,input_label=None):

        feed_dict = self.feed_dict(images=input_data, labels=input_label)

        ops = [self._softmax]
        if input_label is not None:
          ops.append(self._all_plane_accuracy)
          ops.append(self._all_plane_non_bkg_accuracy)

        return sess.run( ops, feed_dict = feed_dict )

    def global_step(self, sess):
        return sess.run(self._global_step)

    def _build_network(self, input_placeholder):

        x = input_placeholder

        # We break up the intial filters into parallel U ResNets
        # The filters are concatenated at the deepest level
        # And then they are split again into the parallel chains

        # print x.get_shape()
        n_planes = self._params['NPLANES']

        x = tf.split(x, n_planes*[1], -1)
        # for p in range(len(x)):
        #     print x[p].get_shape()

        # Initial convolution to get to the correct number of filters:
        for p in range(len(x)):
            x[p] = tf.layers.conv2d(x[p], self._params['N_INITIAL_FILTERS'],
                                    kernel_size=[7, 7],
                                    strides=[1, 1],
                                    padding='same',
                                    use_bias=False,
                                    trainable=self._params['TRAINING'],
                                    name="Conv2DInitial_plane{0}".format(p),
                                    reuse=None)

            # ReLU:
            x[p] = tf.nn.relu(x[p])

        # for p in range(len(x)):
        #     print x[p].get_shape()




        # Need to keep track of the outputs of the residual blocks before downsampling, to feed
        # On the upsampling side

        network_filters = [[] for p in range(len(x))]

        # Begin the process of residual blocks and downsampling:
        for p in xrange(len(x)):
            for i in xrange(self._params['NETWORK_DEPTH']):

                for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                    x[p] = residual_block(x[p], self._params['TRAINING'],
                                          batch_norm=self._params['BATCH_NORM'],
                                          name="resblock_down_plane{0}_{1}_{2}".format(p, i, j))

                network_filters[p].append(x[p])
                x[p] = downsample_block(x[p], self._params['TRAINING'],
                                        batch_norm=self._params['BATCH_NORM'],
                                        name="downsample_plane{0}_{1}".format(p,i))

                # print "Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                #     p=p, i=i, s=x[p].get_shape())

        # print "Reached the deepest layer."

        # Here, concatenate all the planes together before the residual block:
        x = tf.concat(x, axis=-1)
        # print "Shape after concat: " + str(x.get_shape())

        # At the bottom, do another residual block:
        for j in xrange(self._params['RESIDUAL_BLOCKS_DEEPEST_LAYER']):
            x = residual_block(x, self._params['TRAINING'],
                batch_norm=self._params['BATCH_NORM'], name="deepest_block_{0}".format(j))

        # print "Shape after deepest block: " + str(x.get_shape())

        # Need to split the network back into n_planes
        # The deepest block doesn't change the shape, so
        # it's easy to split:
        x = tf.split(x, n_planes, -1)

        # for p in range(len(x)):
        #     print x[p].get_shape()

        # print "Upsampling now."


        # Come back up the network:
        for p in xrange(len(x)):
            for i in xrange(self._params['NETWORK_DEPTH']-1, -1, -1):

                # print "Up start, Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                #     p=p, i=i, s=x[p].get_shape())

                # How many filters to return from upsampling?
                n_filters = network_filters[p][-1].get_shape().as_list()[-1]

                # Upsample:
                x[p] = upsample_block(x[p],
                                      self._params['TRAINING'],
                                      batch_norm=self._params['BATCH_NORM'],
                                      n_output_filters=n_filters,
                                      name="upsample_plane{0}_{1}".format(p,i))


                x[p] = tf.concat([x[p], network_filters[p][-1]],
                                  axis=-1, name='up_concat_plane{0}_{1}'.format(p,i))

                # Remove the recently concated filters:
                network_filters[p].pop()
                with tf.variable_scope("bottleneck_plane{0}_{1}".format(p,i)):
                    # Include a bottleneck to reduce the number of filters after upsampling:
                    x[p] = tf.layers.conv2d(x[p],
                                            n_filters,
                                            kernel_size=[1,1],
                                            strides=[1,1],
                                            padding='same',
                                            activation=None,
                                            use_bias=False,
                                            trainable=self._params['TRAINING'],
                                            name="BottleneckUpsample_plane{0}_{1}".format(p,i))

                    x[p] = tf.nn.relu(x[p])

                # Residual
                for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                    x[p] = residual_block(x[p], self._params['TRAINING'],
                                          batch_norm=self._params['BATCH_NORM'],
                                          name="resblock_up_plane{0}_{1}_{2}".format(p,i, j))

                # print "Up end, Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                #     p=p, i=i, s=x[p].get_shape())


        for p in xrange(len(x)):


            x[p] = residual_block(x[p],
                    self._params['TRAINING'],
                    batch_norm=self._params['BATCH_NORM'],
                    name="FinalResidualBlock_plane{0}".format(p))

            # At this point, we ought to have a network that has the same shape as the initial input, but with more filters.
            # We can use a bottleneck to map it onto the right dimensions:
            x[p] = tf.layers.conv2d(x[p],
                                 self._params['NUM_LABELS'],
                                 kernel_size=[7,7],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 trainable=self._params['TRAINING'],
                                 name="BottleneckConv2D_plane{0}".format(p))
            # print x[p].get_shape()
        # The final activation is softmax across the pixels.  It gets applied in the loss function
#         x = tf.nn.softmax(x)

        return x
