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

"""Iterate through an mxnet symbol definition file that specifies network architecture.
"""
import json


class MxnetParser:

    """Iterator that parses through mxnet json file
     and yields next node in the network.

    Attributes:
        index (int): index of the current node
        nodes (dict): nodes in the mxnet graph

    """

    def __init__(self, symbol_json):
        """ Initialize iterator

        Args:
            symbol_json (str): File name of mxnet json
            file that describes the network.
        """
        # Parse mxnet graph
        with open(symbol_json) as network:
            mxnet_graph = json.load(network)
        self.nodes = mxnet_graph['nodes']
        self.index = 0

    def __iter__(self):
        return self

    def next(self):
        """Next node in the network.

        Returns:
            dict: A dict with parameters of current node in the network.

        Raises:
            StopIteration: When all nodes in the graph have been visited once.
        """
        if self.index == len(self.nodes):
            raise StopIteration()
        node = self.nodes[self.index]
        # null nodes are inputs to the network.
        while node["op"] == "null":
            self.index += 1
            node = self.nodes[self.index]

        inputs = node["inputs"]
        pre_node = [self.nodes[item[0]]["name"]
                    for item in inputs if
                    not self.nodes[item[0]]["name"].endswith('_weight') and
                    not self.nodes[item[0]]["name"].endswith('_bias')]
        self.index += 1
        if "attr" not in node:
            node["attr"] = None
        else:
            node["attr"] = node["attr"]
        return {'type': node["op"],
                'name': node["name"],
                'inputs': pre_node,
                'attr': node["attr"]}
