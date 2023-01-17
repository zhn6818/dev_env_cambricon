#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cnrt_virgo.h"
using namespace CNRT_VIRGO;

class Preprocess{

public:

    void Mat2ChannelFirst(cv::Mat &src, float* p_input);

    void classPreprocess(cv::Mat &img, float* data, int w, int h);

    void detectPreprocess(cv::Mat &img, cv::Mat &to_img, int h, int w);

    float prob_sigmoid(float x);


};
