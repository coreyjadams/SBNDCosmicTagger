
# coding: utf-8

# In[1]:

import os
import ROOT

from larcv import larcv
from matplotlib import pyplot as plt
import numpy
import skimage.measure

# Define global things for saving images:
cmap = plt.cm.winter
norm = plt.Normalize(vmin=0, vmax=50.)


# We have 4 input files: nue, numu, NC, and cosmics only.  We'll process all 4 and get the mask information (as if they were data), as well as the true information for the other 3 files.  With a cut on the number of neutrino pixels and correlations across planes, we can make a crude cosmic rejection tool.

# In[2]:


input_dict = {
    "nueCC"   : "/data/sbnd/processed_files/1k_files/out_nueCC_cosmics_labels.root",
    "numuCC"  : "/data/sbnd/processed_files/1k_files/out_numuCC_cosmics_labels.root",
    "NC"      : "/data/sbnd/processed_files/1k_files/out_NC_cosmics_labels.root",
    "cosmics" : "/data/sbnd/processed_files/1k_files/out_cosmics_only.root",
}


# Below are some useful functions for drawing images, etc:

# In[3]:


def generate_masks(plane, cosmic_scores, neutrino_scores, threshold):
    neutrino_set = neutrino_scores.sparse_tensor_2d(plane)
    cosmic_set   = cosmic_scores.sparse_tensor_2d(plane)

    neutrino_mask = numpy.zeros((320,512))
    cosmic_mask   = numpy.zeros((320,512))

    pos_n_rows = []
    pos_n_cols = []
    meta = neutrino_set.meta()
    for voxel in neutrino_set.as_vector():
        if voxel.value() > threshold:
            neutrino_mask[meta.index_to_row(voxel.id()), meta.index_to_col(voxel.id())] = 1.0

    for voxel in cosmic_set.as_vector():
        if voxel.value() > threshold:
            cosmic_mask[meta.index_to_row(voxel.id()), meta.index_to_col(voxel.id())] = 1.0

    return neutrino_mask, cosmic_mask


# In[4]:


def generate_truth_masks(truth_labels):
    neutrino_label = 1
    cosmic_label   = 2

    cosmic_truth_mask = truth_labels == cosmic_label
    neutrino_truth_mask = truth_labels == neutrino_label
    return truth_labels*neutrino_truth_mask, truth_labels*cosmic_truth_mask


# In[5]:


def upsample_mask(mask, plane):
    if plane == 2:
        trim = (2048 - 1666) / 2
    else:
        trim = (2048 - 1986) / 2

    return numpy.repeat(numpy.repeat(mask, repeats=4, axis=0), repeats=4, axis=1)[:,trim:-trim]


# In[6]:


def display_image(image, name=None, savepath=None):

    if savepath is not None:
        image=cmap(norm(image))
        plt.imsave(savepath, image)
    else:
        figure = plt.figure(figsize=(16,9))
        plt.imshow(image, cmap='winter',vmin=0, vmax=50)
        if name is not None:
            plt.title(name)
        plt.show()



# We want to process each event and determine several things.  For every event:
#  - Number of neutrino pixels, per plane
#  - Number of cosmic pixles, per plane
#  - Central y (time) location and std, per plane
#    - for neutrino pixels only
#
# For only truth files:
#  - True neutrino interaction type (nue, nc, numu)
#  - True neutrino energy
#  - Number of true neutrino pixels above threshold
#  - Sum of adcs of neutrino pixels above threshold
#  - Sum of adcs of cosmic pixels above threshold
#  - Amount of deposited neutrino energy
#  - Accuracy of neutrino prediction, per plane
#  - IoU of neutrino prediction, per plane

# Need to set up an IOManager.  Want to run the cosmic/neutrino pixel labeling:

# In[7]:


def get_io_manager(_file):
    io_manager = larcv.IOManager()
    io_manager.add_in_file(_file)
    io_manager.initialize()
    return io_manager


# In[8]:

# In[9]:


#  - True neutrino interaction type (nue, nc, numu)
#  - True neutrino energy
#  - Number of true neutrino pixels above threshold
#  - Sum of adcs of neutrino pixels above threshold
#  - Sum of adcs of cosmic pixels above threshold
#  - Amount of deposited neutrino energy
#  - Accuracy of neutrino prediction, per plane
#  - IoU of neutrino prediction, per plane

dtypes = {
'type'   : numpy.int8,
'energy' : numpy.float32,
'e_dep'  : numpy.float32,
'n_pix_0': numpy.int64,
'n_pix_1': numpy.int64,
'n_pix_2': numpy.int64,
'c_pix_0': numpy.int64,
'c_pix_1': numpy.int64,
'c_pix_2': numpy.int64,
'n_sum_0': numpy.float32,
'n_sum_1': numpy.float32,
'n_sum_2': numpy.float32,
'c_sum_0': numpy.float32,
'c_sum_1': numpy.float32,
'c_sum_2': numpy.float32,
'n_acc_0': numpy.float32,
'n_acc_1': numpy.float32,
'n_acc_2': numpy.float32,
'n_iou_0': numpy.float32,
'n_iou_1': numpy.float32,
'n_iou_2': numpy.float32,
'n_pix_pred_0' : numpy.int64,
'n_pix_pred_1' : numpy.int64,
'n_pix_pred_2' : numpy.int64,
'c_pix_pred_0' : numpy.int64,
'c_pix_pred_1' : numpy.int64,
'c_pix_pred_2' : numpy.int64,
'n_pix_y_0'    : numpy.float32,
'n_pix_y_1'    : numpy.float32,
'n_pix_y_2'    : numpy.float32,
'n_pix_sigy_0' : numpy.float32,
'n_pix_sigy_1' : numpy.float32,
'n_pix_sigy_2' : numpy.float32,
}


# In[10]:


data_arrs = dict()
for name in input_dict:
    data_arrs[name] = numpy.zeros(1250, dtype={'names': dtypes.keys(), 'formats' : dtypes.values()})
data_arrs['nueCC'][:]['type'] = 0
data_arrs['numuCC'][:]['type'] = 1
data_arrs['NC'][:]['type'] = 2
data_arrs['cosmics'][:]['type'] = 3



# Here is a function that reads in an truth event and returns the necessary information:

# In[12]:


def process_truth_event(_io_manager, entry, _output_array, make_images=False):
    _io_manager.read_entry(entry)

    original_images      = _io_manager.get_data("image2d",   "sbndwire")
    if _output_array[entry]['type'] != 3:
        particle_information = _io_manager.get_data("particle",  "sbndseg")
        particle_seg         = _io_manager.get_data("cluster2d", "sbndseg")
        correct_labels       = _io_manager.get_data("image2d",   "sbnd_cosmicseg")
        neutrino_information = _io_manager.get_data("particle",  "sbndneutrino")
        neutrino = neutrino_information.as_vector().front()

    neutrino_scores      = _io_manager.get_data("sparse2d",  "cosmic")
    cosmic_scores        = _io_manager.get_data("sparse2d",  "neutrino")


#     print "PDG Code: " + str(neutrino.pdg_code())
#     print "Neutrino Energy: " + str(neutrino.energy_init())

    if _output_array[entry]['type'] != 3:
        _output_array[entry]['energy'] = neutrino.energy_init()
#     _output_array[entry]['e_dep'] = neutrino.energy_init()

#     image_name = "/data/sbnd/image_output/3plane_slower_validation_entry{0}".format(entry)

    # This is the information to extract:
    #
    #  - True neutrino interaction type (nue, nc, numu)
    #  - True neutrino energy
    #  - Number of true neutrino pixels above threshold
    #  - Sum of adcs of neutrino pixels above threshold
    #  - Sum of adcs of cosmic pixels above threshold
    #  - Amount of deposited neutrino energy
    #  - Accuracy of neutrino prediction, per plane
    #  - IoU of neutrino prediction, per plane

    for plane in [0,1,2]:

        #These are downsampled masks:
        neutrino_mask, cosmic_mask = generate_masks(plane, cosmic_scores, neutrino_scores, threshold=0.5)

        # Fill in some of the prediction stuff:
        _output_array[entry]['n_pix_pred_{0}'.format(plane)] = numpy.count_nonzero(neutrino_mask)
        _output_array[entry]['c_pix_pred_{0}'.format(plane)] = numpy.count_nonzero(cosmic_mask)

        # Find the average y value of the pixels:
        y_values, x_values = numpy.where(neutrino_mask == 1)
        if len(y_values) != 0:
            _output_array[entry]['n_pix_y_{0}'.format(plane)] = numpy.mean(y_values)
            _output_array[entry]['n_pix_sigy_{0}'.format(plane)] = numpy.std(y_values)
        else:
            _output_array[entry]['n_pix_y_{0}'.format(plane)] = -1.0
            _output_array[entry]['n_pix_sigy_{0}'.format(plane)] = 0.0

        if _output_array[entry]['type'] == 3:
            continue
        # Now fill in the truth based information:




        # These are not downsampled masks:
        neutrino_true, cosmic_true = generate_truth_masks(larcv.as_ndarray(correct_labels.at(plane)))

        _output_array[entry]['n_pix_{0}'.format(plane)] = numpy.count_nonzero(neutrino_true)
        _output_array[entry]['c_pix_{0}'.format(plane)] = numpy.count_nonzero(cosmic_true)


        neutrino_true_ds = numpy.zeros((320, 512))
        cosmic_true_ds = numpy.zeros((320, 512))

        if plane == 2:
            neutrino_true_ds[:,48:-47] = skimage.measure.block_reduce(neutrino_true, (4,4), numpy.max)
            cosmic_true_ds[:,48:-47]   = skimage.measure.block_reduce(cosmic_true, (4,4), numpy.max)
        else:
            neutrino_true_ds[:,8:-7] = skimage.measure.block_reduce(neutrino_true, (4,4), numpy.max)
            cosmic_true_ds[:,8:-7]   = skimage.measure.block_reduce(cosmic_true, (4,4), numpy.max)

        # These are also downsampled masks, but the positive locations:
        neutrino_true_pos = neutrino_true_ds  == 1
        neutrino_mask_pos = neutrino_mask     == 1

        intersection = numpy.logical_and(neutrino_true_pos,  neutrino_mask_pos)
        union        = numpy.logical_or(neutrino_true_pos,  neutrino_mask_pos)
        if numpy.count_nonzero(union) != 0.0:
            iou = 1.0*numpy.count_nonzero(intersection) / numpy.count_nonzero(union)
        else:
            iou = 0.0

        neutrino_locations = neutrino_true_ds == 1
        neutrino_accuracy = numpy.average(neutrino_mask[neutrino_locations] == neutrino_true_ds[neutrino_locations])


        _output_array[entry]['n_iou_{0}'.format(plane)] = iou
        _output_array[entry]['n_acc_{0}'.format(plane)] = neutrino_accuracy


# #         print("Neutrino IoU is {0}".format(iou))


#         print "Neutrino accuracy for plane {0}: {1}".format(plane, neutrino_accuracy)

        raw_image2d    = larcv.as_ndarray(original_images.at(plane))
        neutrino_truth = raw_image2d*neutrino_true
        cosmic_truth   = raw_image2d*cosmic_true

        _output_array[entry]['n_sum_{0}'.format(plane)] = numpy.sum(neutrino_truth)
        _output_array[entry]['c_sum_{0}'.format(plane)] = numpy.sum(cosmic_truth)


        if make_images:
            # This is the upsampled mask to the original resolution.
            # None of this is needed unless making images:
            neutrino_mask  = upsample_mask(neutrino_mask, plane)
            cosmic_mask    = upsample_mask(cosmic_mask, plane)

            neutrino_image = raw_image2d*neutrino_mask
            cosmic_image   = raw_image2d*cosmic_mask



# #         Display the images:
#         display_image(raw_image2d,    name="Raw Event Image, Plane {0}".format(plane),
#                       # savepath=image_name+"_plane{0}_raw_image2d.png".format(plane))
#                      )
#         display_image(neutrino_image, name="Neutrino Prediction, Plane {0}".format(plane),
#                       # savepath=image_name+"_plane{0}_neutrino_image.png".format(plane))
#                      )
# #         display_image(cosmic_image,   name="Cosmic Prediction, Plane {0}".format(plane),
# #                       # savepath=image_name+"_plane{0}_cosmic_image.png".format(plane))
# #                      )
#         display_image(neutrino_truth, name="Neutrino Truth, Plane {0}".format(plane),
#                       # savepath=image_name+"_plane{0}_neutrino_truth.png".format(plane))
#                      )
# #         display_image(cosmic_truth,   name="Cosmic Truth, Plane {0}".format(plane),
# #                       # savepath=image_name+"_plane{0}_cosmic_truth.png".format(plane))
# #                      )



# Loop over the types and fill out the numpy arrays.  We'll save them to file and continue in another notebook.

# In[ ]:


for name in input_dict:
    file_name = "{0}_1250evts.npy".format(name)

    print "Working on {}".format(name)
    if os.path.isfile(file_name):
        continue

    this_io_manager = get_io_manager(input_dict[name])
    for entry in xrange(1250):
        if entry % 25 == 0:
            print "  Entry {}".format(entry)
        process_truth_event(this_io_manager, entry, data_arrs[name])

    numpy.save(file_name, data_arrs[name])
