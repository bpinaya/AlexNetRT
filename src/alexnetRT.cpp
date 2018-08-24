#include <cuda_runtime_api.h>
#include <getopt.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "utils.h"

static Logger gLogger;
static Profiler gProfiler;
using namespace nvinfer1;
using namespace nvcaffeparser1;

static const int BATCH_SIZE = 1;
static const int OUTPUT_SIZE = 1000;

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";

static const int INPUT_H = 227;
static const int INPUT_W = 227;
static const int INPUT_C = 3;

std::string INPUT_IMAGE = "../data/alexnet/dog.ppm";
std::string PROTO_FILE = "../data/alexnet/deploy.prototxt";
std::string WEIGHTS_FILE = "../data/alexnet/bvlc_alexnet.caffemodel";
std::string LABELS_FILE = "../data/alexnet/imagenet-labels.txt";

bool VERBOSE = false;
bool HOTDOG_MODE = false;

void PrintHelp() {
  std::cout << "--input <image.ppm>:             The image input in ppm "
               "format. Defaults to hotdog.ppm\n"
               "--proto <file.prototxt>:         The .prototxt file. Defaults "
               "to deploy.prototxt\n"
               "--weights  <file.caffemodel>:    The weights file for the "
               "network. Defaults to bvlc_alexnet.caffemodel\n"
               "--labels <labels.txt>:           The labels to use. Defautls "
               "to imagenet-labels.txt\n"
               "--verbose                        Outputs extra logs. Defaults "
               "to false.\n"
               "--hotdog                         Silly mode, check if hotdog "
               "or not.\n"
               "--help:                          Show help\n";
  exit(1);
}

void ProcessArgs(int argc, char **argv) {
  const char *const short_opts = "i:p:w:l:v:d:h";
  const option long_opts[] = {{"input", required_argument, nullptr, 'i'},
                              {"proto", required_argument, nullptr, 'p'},
                              {"weights", required_argument, nullptr, 'w'},
                              {"labels", required_argument, nullptr, 'l'},
                              {"verbose", no_argument, nullptr, 'v'},
                              {"hotdog", no_argument, nullptr, 'd'},
                              {"help", no_argument, nullptr, 'h'},
                              {nullptr, no_argument, nullptr, 0}};

  while (true) {
    const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

    if (-1 == opt) break;

    switch (opt) {
      case 'i':
        INPUT_IMAGE = std::string(optarg);
        std::cout << "INPUT_IMAGE set to: " << std::string(optarg) << std::endl;
        break;

      case 'p':
        PROTO_FILE = std::string(optarg);
        std::cout << "PROTO_FILE set to:" << std::string(optarg) << std::endl;
        break;

      case 'w':
        WEIGHTS_FILE = std::string(optarg);
        std::cout << "WEIGHTS_FILE set to: " << std::string(optarg)
                  << std::endl;
        break;

      case 'l':
        LABELS_FILE = std::string(optarg);
        std::cout << "LABELS_FILE file set to: " << std::string(optarg)
                  << std::endl;
        break;

      case 'v':
        VERBOSE = true;
        gLogger.setVerbose();
        std::cout << "VERBOSE set to TRUE\n";
        break;

      case 'd':
        HOTDOG_MODE = true;
        std::cout << "Hotdog or not hotdog?\n";
        break;
      case 'h':
      case '?':
      default:
        PrintHelp();
        break;
    }
  }
}

void AlexnetToTRT(const std::string &deployFile, const std::string &modelFile,
                  const std::string &output, unsigned int maxBatchSize,
                  IHostMemory *&trtModelStream) {
  // Here we import the caffe network, following these 4 steps:
  // 1.- Create the TensorRT Builder and Network
  // 2.- Create the TensorRT parser for the specific format
  // 3.- Use the parser to pase the model and populate the network
  // 4.- Specify the outputs of the network
  // In this function we take as input the path of the .prototxt (deployFile),
  // the path of the .caffemodel (modelFile), the output layer name (output)
  // and the batch size (maxBatchSize), which in our case will be 1 since
  // we eval one image at a time. And we need a space for the serialized
  // network that will be the result (trtModelStream).

  // 1.- Create the TensorRT Builder and Network
  IBuilder *builder = createInferBuilder(gLogger);
  INetworkDefinition *network = builder->createNetwork();

  // 2.- Create the TensorRT parser for the specific format
  ICaffeParser *parser = createCaffeParser();

  DataType modelDataType = DataType::kFLOAT;
  // 3.- Use the parser to pase the model and populate the network
  const IBlobNameToTensor *blobNameToTensor = parser->parse(
      deployFile.c_str(), modelFile.c_str(), *network, modelDataType);
  assert(blobNameToTensor != nullptr);

  // 4.- Specify the outputs of the network
  network->markOutput(*blobNameToTensor->find(output.c_str()));

  // To build an engine in TensorRT there are two steps to follow:
  // 1.- Build the engine using the builder object
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(16 << 20);

  ICudaEngine *engine = builder->buildCudaEngine(*network);
  assert(engine);

  // 2.- Dispense of the network, builder and parser
  network->destroy();
  parser->destroy();

  // Ps. We need to serialize the engine
  trtModelStream = engine->serialize();
  engine->destroy();
  builder->destroy();
  shutdownProtobufLibrary();
}

void InferAndProfile(ICudaEngine *engine, int batchSize) {
  // Run and profile the inference on an image. We are given the engine
  // and the batchSize, which we know is 1 since we do one image at a time.
  // There are 4 steps to perform inference:
  // 1.- Create an execution context to store intermediate activation values.
  // 2.- Use the input and output layer names to get the correct input and
  // output indexes.
  // 3.- Using these indices, set up a buffer array pointing to the input
  // and output buffers on the GPU.
  // 4.- Run the image through the network, while typically asynchronous, we
  // will do synchronous in this case.

  assert(engine->getNbBindings() == 2);
  void *buffers[2];

  // 1.- Create an execution context to store intermediate activation values.
  IExecutionContext *context = engine->createExecutionContext();
  context->setProfiler(&gProfiler);

  // 2.- Use the input and output layer names to get the correct input and
  // output indexes.
  int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

  // allocate GPU buffers
  Dims3 inputDims =
            static_cast<Dims3 &&>(engine->getBindingDimensions(inputIndex)),
        outputDims =
            static_cast<Dims3 &&>(engine->getBindingDimensions(outputIndex));

  // Size in bytes of the input and output.
  // The input size is 618348 bytes, since: 1 x 227 x 227 x 3 x 4 = 618348.
  // Our batch size is one, the image is 227 pixels by 227 pixels in size,
  // there are 3 channels (BGR) and the size of a float is 4 bytes.
  size_t inputSize = batchSize * inputDims.d[0] * inputDims.d[1] *
                     inputDims.d[2] * sizeof(float);
  // The output size is 4000 since: 1 x 1000 x 1 x 1 x 4 = 4000 bytes.
  // Out batch size is one, we have 1000 probabilities in the output, and the
  // rest is 1.
  size_t outputSize = batchSize * outputDims.d[0] * outputDims.d[1] *
                      outputDims.d[2] * sizeof(float);
  if (VERBOSE) {
    std::cout << "inputSize      : " << inputSize << std::endl;
    std::cout << "batchSize      : " << batchSize << std::endl;
    std::cout << "inputDims.d[0] : " << inputDims.d[0] << std::endl;
    std::cout << "inputDims.d[1] : " << inputDims.d[1] << std::endl;
    std::cout << "inputDims.d[2] : " << inputDims.d[2] << std::endl;
    std::cout << "sizeof(float)  : " << sizeof(float) << std::endl;

    std::cout << "outputSize     : " << outputSize << std::endl;
    std::cout << "batchSize      : " << batchSize << std::endl;
    std::cout << "outputDims.d[0]: " << outputDims.d[0] << std::endl;
    std::cout << "outputDims.d[1]: " << outputDims.d[1] << std::endl;
    std::cout << "outputDims.d[2]: " << outputDims.d[2] << std::endl;
    std::cout << "sizeof(float)  : " << sizeof(float) << std::endl;
  }
  // 3.- Using these indices, set up a buffer array pointing to the input
  // and output buffers on the GPU.
  CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
  CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

  uint8_t input[INPUT_H * INPUT_W * INPUT_C];
  ReadPPMImage(INPUT_IMAGE, input, INPUT_H, INPUT_W);

  // Since out image is in uint8_t, we convert it to float and also change
  // the channels from RGB to BGR.
  float *data = new float[INPUT_H * INPUT_W * INPUT_C];
  for (int channel = 0; channel < INPUT_C; ++channel) {
    int pixels = INPUT_H * INPUT_W;
    for (int i = 0; i < pixels; ++i) {
      data[channel * pixels + i] = float(input[i * INPUT_C + 2 - channel]);
    }
  }

  CHECK(
      cudaMemcpy(buffers[inputIndex], data, inputSize, cudaMemcpyHostToDevice));

  // 4.- Run the image through the network, while typically asynchronous, we
  // will do synchronous in this case. We do it the number of times required
  // to profile.
  for (int i = 0; i < TIMING_ITERATIONS; i++)
    context->execute(batchSize, buffers);

  float prob[OUTPUT_SIZE];
  CHECK(cudaMemcpy(prob, buffers[outputIndex], OUTPUT_SIZE * sizeof(float),
                   cudaMemcpyDeviceToHost));

  PrintInference(prob, LABELS_FILE, HOTDOG_MODE);
  // Release the context and buffers
  context->destroy();
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char **argv) {
  std::cout << "\033[1;31mAlexNet deployment with TensorRT." << std::endl;
  std::cout << std::string(50, '*') << "\033[0m" << std::endl;
  ProcessArgs(argc, argv);

  IHostMemory *trtModelStream{nullptr};
  // Import the network using the caffe parser
  AlexnetToTRT(PROTO_FILE, WEIGHTS_FILE, OUTPUT_BLOB_NAME, BATCH_SIZE,
               trtModelStream);
  assert(trtModelStream != nullptr);

  // Create an engine
  IRuntime *infer = createInferRuntime(gLogger);
  assert(infer != nullptr);
  ICudaEngine *engine = infer->deserializeCudaEngine(
      trtModelStream->data(), trtModelStream->size(), nullptr);
  assert(engine != nullptr);

  // Run inference on the engine, also profile it.
  InferAndProfile(engine, BATCH_SIZE);

  // Release the allocated memory
  engine->destroy();
  infer->destroy();
  trtModelStream->destroy();

  // Print the profiled layer times.
  gProfiler.printLayerTimes();
  std::cout << "Done." << std::endl;
  return 0;
}
