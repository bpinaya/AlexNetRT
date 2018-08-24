#include <assert.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "NvInfer.h"

static const int TIMING_ITERATIONS = 1000;
static const int HOTDOG_INDEX = 934;

#define CHECK(status)                       \
  do {                                      \
    auto ret = (status);                    \
    if (ret != 0) {                         \
      std::cout << "Cuda failure: " << ret; \
      abort();                              \
    }                                       \
  } while (0)

class Logger : public nvinfer1::ILogger {
 public:
  Logger() : Logger(Severity::kWARNING) {}
  Logger(Severity severity) : reportableSeverity(severity) {}
  void setVerbose() { reportableSeverity = Severity::kINFO; }
  void log(Severity severity, const char* msg) override {
    if (severity > reportableSeverity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "\033[31mINTERNAL_ERROR:\033[0m ";
        break;
      case Severity::kERROR:
        std::cerr << "\033[31mERROR:\033[0m ";
        break;
      case Severity::kWARNING:
        std::cerr << "\033[33mWARNING:\033[0m ";
        break;
      case Severity::kINFO:
        std::cerr << "\033[34mINFO:\033[0m ";
        break;
      default:
        std::cerr << "\033[37mUNKNOWN:\033[0m ";
        break;
    }
    std::cerr << msg << std::endl;
  }
  Severity reportableSeverity{Severity::kWARNING};
};

struct Profiler : public nvinfer1::IProfiler {
  typedef std::pair<std::string, float> Record;
  std::vector<Record> mProfile;

  virtual void reportLayerTime(const char* layerName, float ms) {
    auto record =
        std::find_if(mProfile.begin(), mProfile.end(),
                     [&](const Record& r) { return r.first == layerName; });
    if (record == mProfile.end())
      mProfile.push_back(std::make_pair(layerName, ms));
    else
      record->second += ms;
  }

  void printLayerTimes() {
    float totalTime = 0;
    printf("\033[1;32mTime of inference per layer:\033[0m\n");
    for (size_t i = 0; i < mProfile.size(); i++) {
      printf("\033[1;32m%-40.40s %4.3fms\033[0m\n", mProfile[i].first.c_str(),
             mProfile[i].second / TIMING_ITERATIONS);
      totalTime += mProfile[i].second;
    }
    printf("\033[1;32mTime over all layers: %4.3fms\033[0m\n",
           totalTime / TIMING_ITERATIONS);
  }
};

inline void ReadPPMImage(const std::string& imageName, uint8_t* buffer, int inH,
                         int inW) {
  std::ifstream infile(imageName, std::ifstream::binary);
  assert(infile.is_open() &&
         "Attempting to read from a file that is not open.");
  std::string magic, h, w, max;
  infile >> magic >> h >> w >> max;
  infile.seekg(1, infile.cur);
  infile.read(reinterpret_cast<char*>(buffer), inH * inW * 3);
}

inline std::vector<std::string> ReadImageNetLabels(
    const std::string& labelFile) {
  std::ifstream label_file(labelFile);
  std::string line;
  std::vector<std::string> labels;
  while (std::getline(label_file, line)) {
    labels.push_back(line);
  }
  return labels;
}

bool SortByProb(const std::pair<int, float>& a,
                const std::pair<int, float>& b) {
  return (a.second > b.second);
}

void PrintInference(float* prob, const std::string& labelFile,
                    bool hotdogMode) {
  std::vector<std::string> labels = ReadImageNetLabels(labelFile);
  typedef std::pair<int, float> inference;
  std::vector<inference> results;
  for (int i = 0; i < 1000; ++i) {
    results.push_back(std::make_pair(i, prob[i]));
  }
  std::sort(results.begin(), results.end(), SortByProb);
  if (hotdogMode) {
    int index = results.at(0).first;
    if (index == HOTDOG_INDEX)
      printf("\033[1;34mHOT DOT!!!\033[0m\n");
    else
      printf("\033[1;34mNOT HOT DOT!!!\033[0m\n");
  } else {
    printf("\033[1;34mResults of inference sorted by confidence:\033[0m\n");
    for (int i = 0; i < 5; ++i) {
      int index = results.at(i).first;
      printf("\033[1;34m%-30.30s %4.2f%%\033[0m\n", labels.at(index).c_str(),
             results.at(i).second * 100);
    }
  }
}