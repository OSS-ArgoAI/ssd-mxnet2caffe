docker run -ti --rm \
-v $PWD:/code \
argoaioss/ssd-mxnet2caffe:cpu /bin/bash \
-c "cd /code; python test_converter.py"

