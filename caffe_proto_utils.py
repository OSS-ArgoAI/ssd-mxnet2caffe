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

from caffe.proto import caffe_pb2
from google.protobuf import text_format
from collections import OrderedDict


class LayerRecord(object):
    """A record which describes layer parameters.
    """

    def __init__(self, layer_def):
        """Reads layer defintion and sets up
        default parameters if none provided and
        coerces convolution parameters to lists instead of ints.
        Sets up layer tops and bottom.

        Args:
            layer_def (caffe layer): Caffe layer definition
        """
        self.layer_def = layer_def
        self.name = layer_def.name
        self.type = layer_def.type
        if layer_def.type in ('Pooling', 'Convolution'):
            if not isinstance(layer_def.convolution_param.kernel_size, list):
                self.filter = [layer_def.convolution_param.kernel_size]
            if len(self.filter) == 1:
                self.filter *= 2

            if not isinstance(layer_def.convolution_param.pad, list):
                self.pad = [layer_def.convolution_param.pad]
            if len(self.pad) == 0:
                self.pad = [0, 0]
            elif len(self.pad) == 1:
                self.pad *= 2

            if not isinstance(layer_def.convolution_param.stride, list):
                self.stride = [layer_def.convolution_param.stride]
            if len(self.stride) == 0:
                self.stride = [1, 1]
            elif len(self.stride) == 1:
                self.stride *= 2
        else:
            self.filter = [0, 0]
            self.pad = [0, 0]
            self.stride = [1, 1]

        self.tops = list(layer_def.top)
        self.bottoms = list(layer_def.bottom)
        self.parents = []
        self.children = []


def read_network_dag(deploy_prototxt):
    """ Reads network structure from caffe_prototxt.

    Args:
        deploy_prototxt (str): name of prototxt to load,

    Returns:
        layer_name_to_record: maps layer name to structure which
          describes in a simple form the layer parameters
        top_to_layers: maps blob name to an ordered list of layers which output that blob
    """

    # load prototxt file
    network_def = caffe_pb2.NetParameter()
    with open(deploy_prototxt, 'r') as proto_file:
        text_format.Merge(str(proto_file.read()), network_def)

    # map layer name to layer record
    layer_name_to_record = OrderedDict()
    for layer_def in network_def.layer:
        if (len(layer_def.include) == 0) or \
           (caffe_pb2.TEST in [item.phase for item in layer_def.include]):
            layer_name_to_record[layer_def.name] = LayerRecord(layer_def)

    top_to_layers = dict()
    for layer in network_def.layer:
        # no specific phase, or TEST phase is specifically asked for
        if (len(layer.include) == 0) or \
                (caffe_pb2.TEST in [item.phase for item in layer.include]):
            for top in layer.top:
                if top not in top_to_layers:
                    top_to_layers[top] = []
                top_to_layers[top].append(layer.name)

    # find parents and children of all layers
    for child_layer_name in layer_name_to_record.keys():
        child_layer_def = layer_name_to_record[child_layer_name]
        for bottom in child_layer_def.bottoms:
            if bottom in top_to_layers:
                for parent_layer_name in top_to_layers[bottom]:
                    if parent_layer_name in layer_name_to_record:
                        parent_layer_def = \
                            layer_name_to_record[parent_layer_name]
                        if parent_layer_def not in child_layer_def.parents:
                            child_layer_def.parents.append(parent_layer_def)
                        if child_layer_def not in parent_layer_def.children:
                            parent_layer_def.children.append(child_layer_def)

    return layer_name_to_record, top_to_layers
