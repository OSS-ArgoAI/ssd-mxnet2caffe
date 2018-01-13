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

from __future__ import absolute_import

import argparse
import caffe
import constants
import logging
import mxnet as mx
import numpy as np
import os
import sys

from caffe import layers
from create_caffe_layers import get_caffe_layer
from parse_mxnet_symbol import MxnetParser
sys.path.append('/incubator-mxnet/example/ssd/tools/caffe_converter')
import caffe_parser  # noqa

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() - %(levelname)-5s ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)
INPUT_DIMS = constants.INPUT_DIMS


class MxNetToCaffe(object):

    """Convert trained model from mxnet to caffe.

    Attributes:
        caffe_prototxt (str): filename of the caffe prototxt
        caffe_weights (str): filename of the caffe weights binary
        epoch (str): mxnet epoch number of model to read weights from
        net (caffe.net): caffe net object that is constructed
        prefix (str): prefix of mxnet model name
    """

    def __init__(self, prefix, epoch, caffe_prototxt=None, caffe_weights=None):
        self.prefix = prefix
        self.epoch = epoch
        self.caffe_prototxt = caffe_prototxt if \
            caffe_prototxt else 'caffe_models/deploy.prototxt'
        self.caffe_weights = caffe_weights if \
            caffe_weights else '{}_{}.caffemodel'.format(prefix, epoch)
        if not os.path.isdir(os.path.dirname(self.caffe_prototxt)):
            os.makedirs(os.path.dirname(self.caffe_prototxt))

        self.caffe_net = None
        self.convert()

    def __parse_network(self):
        """Parse mxnet network and generate corresponding caffe layers.

        """
        # Create caffe network
        caffe_graph = caffe.NetSpec()
        caffe_graph.data = layers.Input(
            input_param={'shape': {'dim': [1, 3, INPUT_DIMS[0], INPUT_DIMS[1]]}})
        # Assign layers from mxnet
        for layer in MxnetParser(self.prefix + '-symbol.json'):
            # Note: name needs to be specified explicitly to reconcile differences in mxnet and caffe norm ops.
            # In caffe norm includes a scaling parameter, in mxnet these are two consecutive ops.
            # So output of the caffe norm op needs to be named based on the scale op name in mxnet.
            caffe_layer = get_caffe_layer(layer, caffe_graph, input_dims=INPUT_DIMS)
            if layer['type'] == 'L2Normalization':
                layer['name'] = 'broadcast_mul0'
            if layer['type'] == 'SoftmaxOutput':
                layer['name'] = 'cls_prob'
            if caffe_layer:
                logger.info("Converting {}".format(layer['type']))
                caffe_graph[layer['name']] = caffe_layer
            else:
                logger.info("Skipping {}".format(layer['type']))
        logger.info('Writing deploy protoxt file to {}.'.format(self.caffe_prototxt))
        with open(self.caffe_prototxt, 'w') as caffe_file:
            caffe_file.write(str(caffe_graph.to_proto()))

    def __assign_weights(self):
        """Assign learnable network weights.
        Network hyper-parameters are assumed to be already set in a previous step.

        Raises:
            ValueError: Unknown batchnorm convention
        """
        # Load caffe prototxt and set up caffe network
        self.caffe_net = caffe.Net(self.caffe_prototxt, caffe.TEST)
        layer_names = self.caffe_net._layer_names
        layers = self.caffe_net.layers
        layer_iter = caffe_parser.layer_iter(layers, layer_names)

        # Load mxnet model
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            self.prefix, self.epoch)
        first_conv = True
        for layer_name, layer_type, layer_blobs in layer_iter:
            if layer_type == 'Normalize':
                assert len(layer_blobs) == 1
                weight_name = [key for key in arg_params.keys()
                               if key.endswith('_scale')][0]
                layer_blobs[0].data[:] = np.squeeze(arg_params[weight_name].asnumpy())

            elif layer_type in ('Convolution', 'InnerProduct'):
                wmat_dim = list(layer_blobs[0].shape)
                weight_name = layer_name + "_weight"
                wmat = arg_params[weight_name].asnumpy().reshape(wmat_dim)
                channels = wmat_dim[1]
                if channels == 3 or channels == 4:  # RGB or RGBA
                    if first_conv:
                        # Swapping RGB in mxnet into BGR of caffe
                        wmat[:, [0, 2], :, :] = wmat[:, [2, 0], :, :]
                        first_conv = False
                assert wmat.flags['C_CONTIGUOUS']
                logger.info('Converting layer {0}, wmat shape = {1}.'.format(
                    layer_name, wmat.shape))
                if weight_name not in arg_params:
                    raise ValueError(weight_name + ' not found in arg_params.')
                layer_blobs[0].data[:] = wmat
                if len(layer_blobs) == 2:
                    bias_name = layer_name + "_bias"
                    if bias_name not in arg_params:
                        raise ValueError(bias_name + ' not found in arg_params.')
                    bias = arg_params[bias_name].asnumpy()
                    assert bias.flags['C_CONTIGUOUS']
                    layer_blobs[1].data[:] = np.squeeze(bias)
                    logger.info(', bias shape = {}.'.format(bias.shape))

            else:
                # Layers with no parameters
                logger.info('\tSkipping layer {} of type {}'.format(
                    layer_name, layer_type))
                assert len(layer_blobs) == 0

    def convert(self):
        """ Converts mxnet model to caffe model.
        Reads through mxnet symbol definition json file and generates corresponding deploy.prototxt.
        Assigns weights from mxnet params file to .caffemodel file.
        """
        self.__parse_network()
        self.__assign_weights()
        logger.info('Saving caffe model in {}'.format(self.caffe_weights))
        self.caffe_net.save(self.caffe_weights)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str,
                        help="prefix of mxnet model")
    parser.add_argument("epoch", type=int,
                        help="epoch number of mxnet model")
    parser.add_argument("caffe_prototxt", type=str,
                        help="filename of caffe deploy prototxt")
    parser.add_argument("caffemodel_name", type=str,
                        help="Name of caffe weights file to save")
    args = parser.parse_args()
    MxNetToCaffe(args.prefix, args.epoch, args.caffe_prototxt, args.caffemodel_name)
