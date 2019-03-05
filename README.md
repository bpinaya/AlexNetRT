# AlexNetRT, a simple TensorRT tutorial

AlexNetRT is a simple deployment of [AlexNet](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf) using [TensorRT](https://developer.nvidia.com/tensorrt). 

Alexnet's authors are Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton.

The purpose of this code is to deploy an already trained (using Caffe with Nvidia Digits) network using TensorRT.

## AlexNet
The architecture of Alexnet can be seen in the `deploy.prototxt` in this [file](/data/alexnet/deploy.prototxt), it's also available on Netscope [here](https://ethereon.github.io/netscope/#/gist/457c00b2539672111ca602a40d7f728f) but you can also get a glimpse in this image:

![alexnet](/data/alexnet/alexnet.svg "Alexnet")

## Building

In order to build this project make sure you have the following requirements: 
- CUDA > 8 (I used 9.0)
- CMake > 3.8  (I used 3.12) - This make it easier to find the Cuda dependencies.
- TensorRT >3 (I used 4.0)
- The original `.caffemodel` that you can get from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) or just run:

```
wget -P data/alexnet/ http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
```
from the root of this project.
- Imagemagick to convert input images to the required format `.ppm`


To build the code do the following:

```bash
git clone https://github.com/bpinaya/AlexNetRT.git alexnetrt && cd alexnetrt
mkdir build && cd build
cmake ..
make
```
## Running
The code itself is documented so you can navigate it and see what each part does, I tried to be as explanatory as possible but also avoided unnecessary details.
You can run it by just calling:
```
./alexnetrt
```
from the build directory but that will launch with the default image (a dog) and run the inference as you can see in this gif:

![alexnet_basic](/data/alexnet/alexnet_simple.gif "Alexnet basic run.")

You can pass it with any image you want as long as it complies with the input format (a 227x227 `.ppm` image), if you want to prepare you image use the `prepare_image.sh` script from the `data/alexnet/` folder as:

```
bash prepare_image.sh image_to_prepare.format
```

Check the cpp file or run `./alexnetrt --help` to check the options, to use a different image as input just run:

```
./alexnetrt --input ../data/alexnet/cat.ppm
```
and you'll see something like this:
![alexnet_cat](/data/alexnet/alexnet_cat.gif "Alexnet with cat.")

## Hotdog not hotdog mode
Finally, making reference to Silicon Valley's TV show you can run it with hotdog mode like this:
```
./alexnetrt --input ../data/alexnet/hotdog.ppm --hotdog
```
![alexnet_hotdog](/data/alexnet/alexnet_hotdog.gif "Alexnet hotdog.")

## TODOs
- Check on FP16 and FP8 precision.
- Run `clang-tidy` and check for errors.

If you have any question or suggestion for improvement let me know creating an issue, PRs are also welcomed.