// #ifndef _SAMPLE_POST_PROCESS_HPP
// #define _SAMPLE_POST_PROCESS_HPP

// #include <map>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <iostream>
// #include <memory>
// #include <string>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cnrt_virgo.h"
using namespace CNRT_VIRGO;

class Postprocess
{
public:

    void post_process(cv::Mat &img, int detect_num, float *out, std::vector<std::string> labels, int dst_h, int dst_w);

    void detect_result(int detect_num, float *out, std::vector<std::vector<float>> output);

};


