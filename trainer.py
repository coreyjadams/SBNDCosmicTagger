import os
import sys
import time

import numpy

# Larcv imports:
import ROOT
from larcv import larcv
larcv.ThreadProcessor
from larcv.dataloader2 import larcv_threadio

import tensorflow as tf

from uresnet import uresnet

class uresnet_trainer(object):

    def __init__(self, config):
        self._config        = config
        self._dataloaders   = dict()
        self._iteration     = 0
        self._batch_metrics = None
        self._output        = None

    def __del__(self):
        self.delete()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.delete()

    def _report(self,metrics,descr):
        msg = ''
        for i,desc in enumerate(descr):
          if not desc: continue
          msg += '%s=%6.6f   ' % (desc,metrics[i])
        msg += '\n'
        sys.stdout.write(msg)
        sys.stdout.flush()

    def delete(self):
        for key, manager in self._dataloaders.iteritems():
            manager.stop_manager()

    def initialize(self):

        dim_data = None

        # Prepare data managers:
        if 'TRAIN_CONFIG' in self._config:
            train_io = larcv_threadio()
            train_io_cfg = {'filler_name' : self._config['TRAIN_CONFIG']['FILLER'],
                            'verbosity'   : self._config['TRAIN_CONFIG']['VERBOSITY'],
                            'filler_cfg'  : self._config['TRAIN_CONFIG']['FILE']}
            train_io.configure(train_io_cfg)
            train_io.start_manager(self._config['MINIBATCH_SIZE'])
            self._dataloaders.update({'train' : train_io})
            self._dataloaders['train'].next(store_entries   = (not self._config['TRAINING']),
                                            store_event_ids = (not self._config['TRAINING']))
            dim_data = self._dataloaders['train'].fetch_data(
                self._config['TRAIN_CONFIG']['KEYWORD_DATA']).dim()

        if 'TEST_CONFIG' in self._config:
            test_io = larcv_threadio()
            test_io_cfg = {'filler_name' : self._config['TEST_CONFIG']['FILLER'],
                            'verbosity'  : self._config['TEST_CONFIG']['VERBOSITY'],
                            'filler_cfg' : self._config['TEST_CONFIG']['FILE']}
            test_io.configure(test_io_cfg)
            test_io.start_manager(self._config['MINIBATCH_SIZE'])
            self._dataloaders.update({'test' : test_io})
            self._dataloaders['test'].next(store_entries   = (not self._config['TRAINING']),
                                           store_event_ids = (not self._config['TRAINING']))
            dim_data = self._dataloaders['test'].fetch_data(
                self._config['TEST_CONFIG']['KEYWORD_DATA']).dim()

        if 'ANA_CONFIG' in self._config:
            ana_io = larcv_threadio()
            ana_io_cfg = {'filler_name' : self._config['ANA_CONFIG']['FILLER'],
                          'verbosity'   : self._config['ANA_CONFIG']['VERBOSITY'],
                          'filler_cfg'  : self._config['ANA_CONFIG']['FILE']}
            ana_io.configure(ana_io_cfg)
            ana_io.start_manager(self._config['MINIBATCH_SIZE'])
            self._dataloaders.update({'ana' : ana_io})
            self._dataloaders['ana'].next(store_entries   = (not self._config['TRAINING']),
                                          store_event_ids = (not self._config['TRAINING']))
            dim_data = self._dataloaders['ana'].fetch_data(
                self._config['ANA_CONFIG']['KEYWORD_DATA']).dim()
            # Output stream (optional)
            if 'OUTPUT' in self._config['ANA_CONFIG']:
                self._output = larcv.IOManager(self._config['ANA_CONFIG']['OUTPUT'])
                self._output.initialize()



        # Net construction:
        self._net = uresnet(self._config)
        self._net.construct_network(dims=dim_data)

        #
        # Network variable initialization
        #

        # Configure global process (session, summary, etc.)
        # Initialize variables
        self._sess = tf.Session()
        self._writer = tf.summary.FileWriter(self._config['LOGDIR'] + '/train/')
        self._saver = tf.train.Saver()

        if 'TEST_CONFIG' in self._config:
            self._writer_test = tf.summary.FileWriter(self._config['LOGDIR'] + '/test/')

        if not self._config['RESTORE']:
                self._sess.run(tf.global_variables_initializer())
                self._writer.add_graph(self._sess.graph)
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self._config['LOGDIR']+"/train/checkpoints/")
            print "Restoring model from {}".format(latest_checkpoint)
            self._saver.restore(self._sess, latest_checkpoint)


    def train_step(self):

        self._iteration = self._net.global_step(self._sess)
        report_step  = self._iteration % self._config['REPORT_ITERATION'] == 0
        summary_step = 'SUMMARY_ITERATION' in self._config and (self._iteration % self._config['SUMMARY_ITERATION']) == 0
        checkpt_step = 'SAVE_ITERATION' in self._config and (self._iteration % self._config['SAVE_ITERATION']) == 0

        # Nullify the gradients
        self._net.zero_gradients(self._sess)

        # Loop over minibatches
        for j in xrange(self._config['N_MINIBATCH']):
            minibatch_data   = self._dataloaders['train'].fetch_data(
                self._config['TRAIN_CONFIG']['KEYWORD_DATA']).data()
            # reshape right here:
            minibatch_data = numpy.reshape(minibatch_data,
                self._dataloaders['train'].fetch_data(
                    self._config['TRAIN_CONFIG']['KEYWORD_DATA']).dim()
                )
            minibatch_label  = self._dataloaders['train'].fetch_data(
                self._config['TRAIN_CONFIG']['KEYWORD_LABEL']).data()
            minibatch_label = numpy.reshape(
                minibatch_label, self._dataloaders['train'].fetch_data(
                    self._config['TRAIN_CONFIG']['KEYWORD_LABEL']).dim()
                )
            minibatch_weight = None
            if self._config['BALANCE_LOSS']:
                if 'KEYWORD_WEIGHT' in self._config['TRAIN_CONFIG']:
                    minibatch_weight = self._dataloaders['train'].fetch_data(
                        self._config['TRAIN_CONFIG']['KEYWORD_WEIGHT']).data()
                    minibatch_weight = numpy.reshape(
                        minibatch_weight, self._dataloaders['train'].fetch_data(
                            self._config['TRAIN_CONFIG']['KEYWORD_WEIGHT']).dim()
                        )
                else:
                    minibatch_weight = self.compute_weights(minibatch_label)
            # perform per-event normalization


            # compute gradients
            res,doc = self._net.accum_gradients(sess         = self._sess,
                                                input_data   = minibatch_data,
                                                input_label  = minibatch_label,
                                                input_weight = minibatch_weight)

            self._dataloaders['train'].next(store_entries   = (not self._config['TRAINING']),
                                            store_event_ids = (not self._config['TRAINING']))

            if self._batch_metrics is None:
                self._batch_metrics = numpy.zeros((self._config['N_MINIBATCH'],len(res)-1),dtype=numpy.float32)
                self._descr_metrics = doc[1:]

            self._batch_metrics[j,:] = res[1:]

        # update
        self._net.apply_gradients(self._sess)


        # read-in test data set if needed
        (test_data, test_label, test_weight) = (None,None,None)
        if (report_step or summary_step) and 'TEST_CONFIG' in self._config:
            self._dataloaders['test'].next()
            test_data   = self._dataloaders['test'].fetch_data(
                self._config['TEST_CONFIG']['KEYWORD_DATA']).data()
            test_label  = self._dataloaders['test'].fetch_data(
                self._config['TEST_CONFIG']['KEYWORD_DATA']).data()
            test_weight = None
            if self._config['BALANCE_LOSS']:
                if 'KEYWORD_WEIGHT' in self._config['TEST_CONFIG']:
                    minibatch_weight = self._dataloaders.fetch_data(
                        self._config['TEST_CONFIG']['KEYWORD_WEIGHT']).data()
                else:
                    test_weight = self.compute_weights(test_label)

        # Report
        if report_step:
            sys.stdout.write('@ iteration {}\n'.format(self._iteration))
            sys.stdout.write('Train set: ')
            self._report(numpy.mean(self._batch_metrics,axis=0),self._descr_metrics)
            if 'test' in self._dataloaders:
                res,doc = self._net.run_test(self._sess, test_data, test_label, test_weight)
                sys.stdout.write('Test set: ')
                self._report(res,doc)

        # Save log
        if summary_step:
            # Run summary
            self._writer.add_summary(self._net.make_summary(self._sess,
                                                            minibatch_data,
                                                            minibatch_label,
                                                            minibatch_weight),
                                     self._iteration)
            if 'TEST_CONFIG' in self._config:
                self._writer_test.add_summary(self._net.make_summary(self._sess, test_data, test_label, test_weight),
                                              self._iteration)

        # Save snapshot
        if checkpt_step:
            # Save snapshot
            ssf_path = self._saver.save(self._sess,
                self._config['LOGDIR']+"/train/checkpoints/save",
                global_step=self._iteration)
            sys.stdout.write('saved @ ' + str(ssf_path) + '\n')
            sys.stdout.flush()

    def ana(self, input_data, input_label):
        return  self._net.inference(sess        = self._sess,
                                    input_data  = input_data,
                                    input_label = input_label)
        pass


    def ana_step(self):

        # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
        batch_data   = self._dataloaders['ana'].fetch_data(
            self._config['ANA_CONFIG']['KEYWORD_DATA']).data()
        batch_label  = self._dataloaders['ana'].fetch_data(
            self._config['ANA_CONFIG']['KEYWORD_LABEL']).data()
        batch_weight = None
        softmax,acc_all,acc_nonzero = self.ana(input_data  = batch_data,
                                               input_label = batch_label)
        if self._output:
          for entry in xrange(len(softmax)):
            self._output.read_entry(entry)
            data  = np.array(batch_data[entry]).reshape(softmax.shape[1:-1])
            entries   = self._input_main.fetch_entries()
            event_ids = self._input_main.fetch_event_ids()

          for entry in xrange(len(softmax)):
            self._output.read_entry(entries[entry])
            data  = np.array(batch_data[entry]).reshape(softmax.shape[1:-1])
            label = np.array(batch_label[entry]).reshape(softmax.shape[1:-1])

            shower_score, track_score = (None,None)
            data_2d = len(softmax.shape) == 4
            # 3D case
            if data_2d:
              shower_score = softmax[entry,:,:,1]
              track_score  = softmax[entry,:,:,2]
            else:
              shower_score = softmax[entry,:,:,:,1]
              track_score  = softmax[entry,:,:,:,2]

            sum_score = shower_score + track_score
            shower_score = shower_score / sum_score
            track_score  = track_score  / sum_score

            ssnet_result = (shower_score > track_score).astype(np.float32) + (track_score >= shower_score).astype(np.float32) * 2.0
            nonzero_map  = (data > 1.0).astype(np.int32)
            ssnet_result = (ssnet_result * nonzero_map).astype(np.float32)

            if data_2d:
              larcv_data = self._output.get_data("image2d","data")
              larcv_out  = self._output.get_data("sparse2d","ssnet")
              vs = larcv.as_image2d(ssnet_result)
              sparse3d.set(vs,data.meta())
            else:
              larcv_data = self._output.get_data("sparse3d","data")
              larcv_out  = self._output.get_data("sparse3d","ssnet")
              vs = larcv.as_tensor3d(ssnet_result)
              sparse3d.set(vs,data.meta())
            self._output.save_entry()

        self._input_main.next(store_entries   = (not self._cfg.TRAIN),
                              store_event_ids = (not self._cfg.TRAIN))



    def batch_process(self):

        # Run iterations
        for i in xrange(self._config['ITERATIONS']):
            if self._config['TRAINING'] and self._iteration >= self._config['ITERATIONS']:
                print('Finished training (iteration %d)' % self._iteration)
                break

            # Start IO thread for the next batch while we train the network
            if self._config['TRAINING']:
                self.train_step()
            else:
                self.ana_step()


    def compute_weights(self, labels):
        # Take the labels, and compute the per-label weight

        # Prepare output weights:
        weights = numpy.zeros(labels.shape)

        i = 0
        for batch in labels:
            # First, figure out what the labels are and how many of each:
            values, counts = numpy.unique(batch, return_counts=True)

            n_pixels = numpy.sum(counts)
            for value, count in zip(values, counts):
                weight = 1.0*(n_pixels - count) / n_pixels
                mask = labels[i] == value
                weights[i, mask] += weight

            weights[i] *= 1. / numpy.sum(weights[i])
            i += 1

        # Normalize the weights to sum to 1 for each event:
        return weights