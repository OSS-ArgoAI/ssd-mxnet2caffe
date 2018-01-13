docker run -ti --rm \
-v $PWD:/code \
argoaioss/ssd-mxnet2caffe:cpu /bin/bash \
-c "cd /code; coverage run --source ./ test_converter.py; coverage html"

