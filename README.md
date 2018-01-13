![ARGO](https://github.com/OSS-ArgoAI/ssd-mxnet2caffe/blob/master/assets/ARGO_logo.png)

[![Build Status](https://travis-ci.com/argoai/ssd-mxnet2caffe.svg?token=HpRyp8wyHLUjnsWzHene&branch=master)](https://travis-ci.com/argoai/ssd-mxnet2caffe) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


 
SSD-mxnet2caffe
================

Tool to convert a pre-trained [single shot object detection](https://arxiv.org/abs/1512.02325) model from MxNet to Caffe.

Input
-----
This tool has been tested with the ssd [model](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.5-beta/vgg16_ssd_512_voc0712_trainval.zip) trained in MxNet for VOC0712 challenge with the VGG16 front-end.

Environment
-----------
A dockerfile that creates the environment with Mxnet and Caffe can be found in the docker folder.

* Mxnet version: 0.12.1
* Caffe version: 1.0 with custom layers implemented in [Wei Liu's ssd branch of Caffe](https://github.com/weiliu89/caffe/tree/ssd).

Tests
-----
To run the tests launch the run_tests.sh script.
Sample results are saved in result.png

![](assets/result.png)




