# Copyright 2018 Argo AI, LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run tests on the converter.
"""
from __future__ import absolute_import

import matplotlib
matplotlib.use('agg')  # noqa
import logging
import mxnet as mx
import numpy as np
import os
import unittest
import urllib
import zipfile

from caffe_proto_utils import read_network_dag
from matplotlib import pyplot
from mxnet_to_caffe import MxNetToCaffe
from utilities import bfs
from utilities import print_progress
from utilities import process_layer_output
from utilities import process_layer_parameters
from utilities import read_img
from utilities import run_inference_caffe
from utilities import run_inference_mxnet

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() - %(levelname)-5s ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConverter(unittest.TestCase):

    """ Run tests on the converter.
    """

    @classmethod
    def setUpClass(cls,
                   image_url='./assets/dog.jpg',
                   mx_prefix='mxnet_models/vgg16_ssd_300_voc0712_trainval/ssd_300',
                   mx_epoch=0,
                   caffe_mean=[0, 0, 0],
                   caffe_prototxt='caffe_models/deploy.prototxt',
                   caffe_weights='caffe_models/weights.caffemodel'):
        cls.image_url = image_url
        cls.caffe_mean = caffe_mean
        cls.caffe_prototxt = caffe_prototxt
        cls.caffe_weights = caffe_weights
        # Download mxnet model if not found
        filename = mx_prefix + '-symbol.json'
        link = "https://goo.gl/fNonDd"
        model_dir = os.path.dirname(mx_prefix)
        if not os.path.isfile(filename):
            if not os.path.isdir(model_dir):
                logger.info('Creating: ' + model_dir)
                os.makedirs(model_dir)
            logger.info('Downloading mxnet model to test converter')
        try:
            urllib.urlretrieve(link, model_dir + '/model.zip', reporthook=print_progress)
            with zipfile.ZipFile(model_dir + '/model.zip', 'r') as zip_ref:
                zip_ref.extractall('mxnet_models/')
        except Exception as inst:
            logger.info(inst)

        # Load mxnet model
        logger.info('Loading mxnet model')
        cls.sym, cls.arg_params, cls.aux_params = mx.model.load_checkpoint(
            mx_prefix, mx_epoch)

        # Convert model to caffe
        logger.info('Converting to caffe model')
        cls.convertor = MxNetToCaffe(mx_prefix, mx_epoch, caffe_prototxt, caffe_weights)
        cls.layer_to_record, cls.top_to_layers = read_network_dag(cls.caffe_prototxt)

    @classmethod
    def tearDownClass(cls):
        logger.info('Cleaning up..')
        os.remove(cls.caffe_prototxt)
        os.remove(cls.caffe_weights)
        logger.info('Deleted files {}, {}.'.format(cls.caffe_prototxt, cls.caffe_weights))

    def test_parameters_layerwise(self):
        """Test converted model layer-wise
        """

        logger.info('\n***** Network Parameters '.ljust(140, '*'))
        log_format = '  {0:>40}  {1:>30}  {2:>8}  {3:>10}'
        title_format = '  {0:>30}  {1:>30}  {2:>8}  {3:>15}'
        logger.info(title_format.format('CAFFE', 'MXNET',
                                        'Mean(diff)', 'Max(diff)'))
        first_layer_name = self.layer_to_record.keys()[0]
        bfs(self.layer_to_record[first_layer_name],
            process_layer_parameters, logger,
            self.convertor.caffe_net, self.arg_params,
            self.aux_params, log_format)

    def test_activations_layerwise(self):
        """Test activations for each layer
        """
        # Set up input
        # swap channels from Caffe BGR to RGB
        self.caffe_mean = self.caffe_mean[::-1]
        image = read_img(self.image_url, mean=self.caffe_mean)

        # Run inference through mxnet and caffe
        mxnet_network = run_inference_mxnet(self.sym, self.arg_params,
                                            self.aux_params, image, gpu=-1)
        caffe_network = run_inference_caffe(self.convertor.caffe_net, image)
        logger.info('\n***** Network Outputs '.ljust(140, '*'))
        log_format = '  {0:>40}  {1:>30}  {2:>20}  {3:>20}'
        title_format = '  {0:>30}  {1:>30}  {2:>8}  {3:>15}'
        logger.info(title_format.format('CAFFE', 'MXNET',
                                        'Activation-Mean(diff)',
                                        'Activation-Max(diff)'))
        for caffe_blob_name in caffe_network.blobs.keys():
            process_layer_output(mxnet_network, caffe_network,
                                 caffe_blob_name, logger, log_format,
                                 self.top_to_layers)

    def test_final_result(self):
        """Plot final detections on test image
        """

        # swap channels from Caffe BGR to RGB
        self.caffe_mean = self.caffe_mean[::-1]
        image = read_img(self.image_url, mean=self.caffe_mean)

        # Run inference through mxnet and caffe
        mxnet_exe = run_inference_mxnet(self.sym, self.arg_params,
                                        self.aux_params, image, gpu=-1)
        caffe_network = run_inference_caffe(self.convertor.caffe_net, image)
        mx_output = mxnet_exe.output_dict['det_out_output'].asnumpy()
        mx_label = mx_output[0, np.argmax(mx_output[0, :, 1]), 0]
        mx_conf = mx_output[0, np.argmax(mx_output[0, :, 1]), 1]
        pyplot.ion()
        pyplot.figure()
        pyplot.subplot(2, 1, 1)
        pyplot.title('MXNET RESULTS')
        im = pyplot.imshow(np.transpose(image[0].astype(np.uint8), (1, 2, 0)))
        pyplot.tight_layout()
        pyplot.xticks([])
        pyplot.yticks([])
        xmin, ymin, xmax, ymax = mx_output[0,
                                           np.argmax(mx_output[0, :, 1]), 2:]
        img_width, img_height = image.shape[2:]
        xmin *= img_width
        xmax *= img_width
        ymin *= img_height
        ymax *= img_height
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        im.axes.add_patch(pyplot.Rectangle(
            *coords, fill=False, edgecolor='r', linewidth=2))
        pyplot.subplot(2, 1, 2)
        pyplot.title('CAFFE RESULTS')
        im = pyplot.imshow(np.transpose(image[0].astype(np.uint8), (1, 2, 0)))
        caffe_output = caffe_network.blobs['detection'].data

        det_conf = caffe_output[0, 0, :, 2]
        xmin, ymin, xmax, ymax = caffe_output[0, 0, np.argmax(det_conf), -4:]
        caffe_label = caffe_output[0, 0, np.argmax(
            det_conf), 1] - 1  # Subtract background label id
        caffe_conf = caffe_output[0, 0, np.argmax(
            det_conf), 2]

        assert caffe_label == mx_label
        assert abs(caffe_conf - mx_conf) < 1e-6

        xmin *= img_width
        xmax *= img_width
        ymin *= img_height
        ymax *= img_height
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1

        im.axes.add_patch(pyplot.Rectangle(
            *coords, fill=False, edgecolor='g', linewidth=2))
        pyplot.tight_layout()
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.savefig('result.png')
        logger.info('Saving predictions in result.png.')


if __name__ == "__main__":
    unittest.main()
