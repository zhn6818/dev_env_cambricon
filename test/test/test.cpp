#include <iostream>
#include "cnrt_virgo.h"
using namespace ASCEND_VIRGO;
int main()
{
    std::shared_ptr<Classify> dffg = std::make_shared<Classify>("/data1/qwy/code/magicmind_cloud-master/buildin/cv/classification/resnet50_onnx/data/models/resnet50_onnx_model_force_float32_true_1", "", 0);

    cv::Mat img = cv::imread("/data1/qwy/dataset/ILSVRC2012/ILSVRC2012_val_00000001.JPEG");
    std::vector<cv::Mat> sdf;
    sdf.push_back(img);
    std::vector<Predictioin> df;
    dffg->process(sdf, df);
    return 0;
}