FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    build-essential git libatlas-base-dev libopencv-dev python-opencv \
    libcurl4-openssl-dev libgtest-dev cmake wget unzip
 
RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib

RUN git clone -b 0.12.1 --recursive https://github.com/apache/incubator-mxnet.git /incubator-mxnet
ENV BUILD_OPTS "USE_CUDA=0 USE_CUDNN=0 USE_OPENCV=1"
RUN cd /incubator-mxnet && \
    make -j$(nproc) $BUILD_OPTS

RUN apt-get update && apt-get install -y python-dev python3-dev
RUN cd /tmp && wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && python2 get-pip.py
RUN pip2 install nose pylint numpy nose-timer requests
RUN pip3 install nose pylint numpy nose-timer requests
ENV PYTHONPATH=/incubator-mxnet/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
    python-tk \
    python-matplotlib \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
RUN git clone -b ssd --depth 1 https://github.com/weiliu89/caffe.git . && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=0 -DUSE_NCCL=0 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
RUN pip install coverage
