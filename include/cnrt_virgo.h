#pragma once

#include <iostream>
#include <memory>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <ostream>
#include <sstream>
#include "cnrt.h"
#include <mm_runtime.h>
#include <opencv2/opencv.hpp>

using namespace magicmind;

namespace CNRT_VIRGO
{
    // static float prob_sigmoid(float x) { return (1 / (1 + exp(-x))); }
    typedef std::pair<std::string, float> Predictioin;
    class ClassifyPrivate;
    class Classify
    {
    public:
        Classify(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath =  "/data1/qwy/code/yolov7config/yolov7_detect_config/CDdetect_test.json");
        ~Classify();
        void process(std::vector<cv::Mat> &vecMat, std::vector<Predictioin> &result);
        size_t GetBatch();
        size_t GetInputSize();

    private:
        std::shared_ptr<ClassifyPrivate> m_pHandlerClassifyPrivate;
        // std::shared_ptr<ClassifyDvpp> m_pHandlerClassifyDvpp;
    };

    

    //yolov7 detect
    typedef std::pair<std::string, float> Predictioin;
    class DetectPrivate;
    class Detect
    {
    public:
        Detect(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath =  " ");
        ~Detect();
        void process(std::vector<cv::Mat> &vecMat, std::vector<Predictioin> &result);
        size_t GetBatch();
        size_t GetInputSize();

    private:
        std::shared_ptr<DetectPrivate> m_pHandle;
    };

    struct DetectedObject
    {
    int object_class;
    float prob;
    cv::Rect bounding_box;

    // DetectedObject()
    //     : object_class(-1), prob(0.), bounding_box(cv::Rect(0, 0, 0, 0)) {}
    // DetectedObject(int object_class, float prob, cv::Rect bb)
    //     : object_class(object_class), prob(prob), bounding_box(bb) {}
    };
};