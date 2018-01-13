# Copyright 2018 Argo AI, LLC

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import caffe
import cv2
import mxnet as mx
import numpy as np
import sys
import time
import re
import urllib


def print_progress(count, block_size, total_size):
    """Prints progress status of download.

    Args:
        count (int): Count of blocks transferred so far
        block_size (int): Block size in bytes
        total_size (int): Total size of the file

    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\rDownloading mxnet model...%d%%, %d MB, %d KB/s, %d seconds" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def change_device(arg_params, aux_params, ctx):
    """Changes device of given mxnet arguments

    Args:
        arg_params (dict): arguments
        aux_params (dict): auxiliary parameters
        ctx (mx.cpu or mx.gpu): new device context

    Returns:
        dicts: arguments and auxiliary parameters on new device
    """
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs


def read_img(img_path, image_dims=(300, 300), mean=None):
    """Sets up input data for the tests.

    Args:
        img_path (str): Path to file or url to download
        image_dims (None, optional): Image dims to resize to
        mean (None, optional): mean value to subtract

    Returns:
        np.array: Image in nchw format.
    """
    filename = img_path.split("/")[-1]
    if img_path.startswith('http'):
        urllib.urlretrieve(img_path, filename)
        img = cv2.imread(filename)
    else:
        img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image_dims is not None:
        # resize to image_dims to fit model
        img = cv2.resize(img, image_dims)
    img = np.rollaxis(img, 2)  # change to (c, h, w) order
    img = img[np.newaxis, :]  # extend to (n, c, h, w)
    if mean is not None:
        mean = np.array(mean)
    if mean.shape == (3,):
        mean = mean[np.newaxis, :, np.newaxis,
                    np.newaxis]  # extend to (n, c, 1, 1)

    return img.astype(np.float32) - mean  # subtract mean


def run_inference_mxnet(sym, arg_params, aux_params, image, gpu):
    """Run inference in mxnet

    Args:
        sym: Pretrained network symbol
        arg_params: Argument parameters of the pretrained model
        aux_params: Auxilliary parameters of the pretrained model
        image (np array): Input to the network
        gpu (int): gpu id

    Returns:
        mxnet.exe: mxnet executible
    """
    if gpu < 0:
        ctx = mx.cpu(0)
    else:
        ctx = mx.gpu(gpu)

    arg_params, aux_params = change_device(arg_params, aux_params, ctx)
    arg_params["data"] = mx.nd.array(image, ctx)
    arg_params["label"] = mx.nd.empty((1, 21, 5), ctx)
    sym = sym.get_internals()
    exe = sym.bind(ctx, arg_params, args_grad=None,
                   grad_req="null", aux_states=aux_params)
    exe.forward(is_train=False)
    return exe


def run_inference_caffe(caffe_net, image):
    """Run inference in caffe

    Args:
        caffe_net: caffe network
        image (np array): input to the network

    Returns:
        caffe_net after forward propogation of input
    """
    caffe.set_mode_cpu()

    img_bgr = image[:, :: -1, :, :]
    caffe_net.blobs['data'].reshape(*img_bgr.shape)
    caffe_net.blobs['data'].data[...] = img_bgr
    caffe_net.forward()

    return caffe_net


def bfs(root_node, process_node, logger,
        caffe_net, arg_params, aux_params, log_format):
    """ Implementation of breadth-first search (BFS) on caffe network DAG

    Args:
        root_node : root node of caffe network DAG
        process_node: function to run on each node
        logger: logger to log
        caffe_net: caffe network
        arg_params: Argument parameters of the pretrained model
        aux_params: Auxilliary parameters of the pretrained model
        log_format: log format
    """

    from collections import deque

    seen_nodes = set()
    next_nodes = deque()

    seen_nodes.add(root_node)
    next_nodes.append(root_node)

    while next_nodes:
        current_node = next_nodes.popleft()

        # process current node
        process_node(current_node, logger, caffe_net,
                     arg_params, aux_params, log_format)

        for child_node in current_node.children:
            if child_node not in seen_nodes:
                seen_nodes.add(child_node)
                next_nodes.append(child_node)


def compare_blob(caf_blob, mx_blob, caf_name, mx_name,
                 log_format, logger):
    """Compare tensors generated by mxnet and caffe

    Args:
        caf_blob: Caffe blob
        mx_blob: Mxnet tensor
        caf_name: Name of caffe blob
        mx_name: Name of mxnet blob
        log_format: string format of log
        logger: logger
    """
    assert mx_blob.shape == caf_blob.shape, \
        'Caffe blob-{} and mxnet blob-{} are not the same shape {} vs {}'.format(
            caf_name, mx_name, caf_blob.shape, mx_blob.shape)
    diff = np.abs(mx_blob - caf_blob)
    diff_mean = diff.mean()
    diff_max = diff.max()
    logger.info(log_format.format(caf_name, mx_name,
                                  '%4.5f' % diff_mean,
                                  '%4.5f' % diff_max))
    assert diff_mean < 1e-03
    assert diff_max < 1e-02


def process_layer_parameters(layer, logger, caffe_net,
                             arg_params, aux_params, log_format):
    """Check layer weights translated to caffe

    Args:
        layer : layer generated by caffe dag
        logger: logger to log
        caffe_net: caffe network
        arg_params: Argument parameters of the pretrained model
        aux_params: Auxilliary parameters of the pretrained model
        log_format: log format

    """
    logger.debug('Processing layer %s of type %s.', layer.name, layer.type)
    normalized_layer_name = re.sub('[-/]', '_', layer.name)
    # handle weight and bias of convolution and fully-connected layers
    if layer.name in caffe_net.params and \
        layer.type in ['Convolution', 'InnerProduct',
                       'Deconvolution']:
        has_bias = len(caffe_net.params[layer.name]) > 1
        mx_name_weight = '{}_weight'.format(normalized_layer_name)
        mx_beta = arg_params[mx_name_weight].asnumpy()

        first_conv = True
        # first convolution should change from BGR to RGB
        if layer.type == 'Convolution' and first_conv:
            first_conv = False
            # if RGB or RGBA
            if mx_beta.shape[1] == 3 or mx_beta.shape[1] == 4:
                # Swapping BGR of caffe into RGB in mxnet
                mx_beta[:, [0, 2], :, :] = mx_beta[:, [2, 0], :, :]

        caf_beta = caffe_net.params[layer.name][0].data
        compare_blob(caf_beta, mx_beta, layer.name,
                     mx_name_weight, log_format, logger)
        if has_bias:
            mx_name_bias = '{}_bias'.format(normalized_layer_name)
            mx_gamma = arg_params[mx_name_bias].asnumpy()
            caf_gamma = caffe_net.params[layer.name][1].data
            compare_blob(caf_gamma, mx_gamma, layer.name,
                         mx_name_bias, log_format, logger)
    else:
        logger.debug('No paramters to check for layer %s of type %s',
                     layer.name, layer.type)


def process_layer_output(mxnet_exe, caffe_net, caffe_blob_name,
                         logger, log_format, top_to_layers):
    """Check layer-wise activations.

    Args:
        mxnet_exe: mxnet executible
        caffe_net: caffe network
        caffe_blob_name: name of the caffe blob
        logger: logger
        log_format: string format of the log
        top_to_layers: dict that maps top to layer name
    """
    logger.debug('processing blob %s', caffe_blob_name)

    # skip blobs not originating from actual layers,
    # e.g. artificial split layers added by caffe
    if caffe_blob_name not in top_to_layers:
        return
    if caffe_blob_name.endswith('detection'):
        # Skip last layer since outputs are different format
        return
    caf_blob = caffe_net.blobs[caffe_blob_name].data
    # data should change from BGR to RGB
    if caffe_blob_name == 'data':
        # if RGB or RGBA
        if caf_blob.shape[1] == 3 or caf_blob.shape[1] == 4:
            # Swapping BGR of caffe into RGB in mxnet
            caf_blob[:, [0, 2], :, :] = caf_blob[:, [2, 0], :, :]
        mx_name = 'data'
    else:
        last_layer_name = top_to_layers[caffe_blob_name][-1]
        normalized_last_layer_name = re.sub('[-/]', '_', last_layer_name)
        mx_name = '{}_output'.format(normalized_last_layer_name)
    if mx_name not in mxnet_exe.output_dict:
        logger.warn('mxnet blob %s is missing for caffe blob %s', mx_name, caffe_blob_name)
        return
    mx_blob = mxnet_exe.output_dict[mx_name].asnumpy()

    if caf_blob.shape != mx_blob.shape:
        if 'cls_prob' in caffe_blob_name:
            mx_blob = mx_blob.transpose((0, 2, 1)).reshape(caf_blob.shape)
        elif 'anchors' in caffe_blob_name or 'concat' in caffe_blob_name or 'flatten' in caffe_blob_name:
            caf_blob = caf_blob[0, 0, :].reshape(mx_blob.shape)
        else:
            logger.error('Shape mismatch in {0} and {1}'.format(caffe_blob_name, mx_name))

    compare_blob(caf_blob, mx_blob, caffe_blob_name,
                 mx_name, log_format, logger)
