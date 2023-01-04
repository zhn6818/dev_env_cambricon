#include <iostream>
#include "cnrt_virgo.h"
using namespace ASCEND_VIRGO;
int main()
{
    std::shared_ptr<Classify> dffg = std::make_shared<Classify>("/data1/qwy/code/magicmind_cloud-master/buildin/cv/classification/resnet50_onnx/data/models/resnet50_onnx_model_force_float32_true_1", "", 0);
    std::vector<cv::Mat> sdf;

    std::vector<Predictioin> df;
    dffg->process(sdf, df);
    return 0;
}