#include <iostream>
#include "cnrt_virgo.h"
#include <dirent.h>
using namespace CNRT_VIRGO;
int main()
{
    std::shared_ptr<Detect> dffg = std::make_shared<Detect>("/data1/qwy/code/magicmind_cloud-master2/buildin/cv/detection/yolov7_pytorch/data/models/yolov7_pytorch_model_qint8_mixed_float16_true", 
                                                            "/data1/qwy/code/magicmind_cloud-master2/buildin/cv/utils/coco.names", 0);
    // std::shared_ptr<Classify> dffg = std::make_shared<Classify>("/usr/local/neuware/samples/magicmind/mm_build/model/resnet18.model", 
    //                                                             "/data1/qwy/labels/commonworker.names", 
    //                                                             0);
    
    // *********************batch == 1***********************
    cv::Mat img = cv::imread("/data1/qwy/dataset/coco/test/000000000139.jpg");
    std::vector<cv::Mat> sdf;
    sdf.push_back(img);
    std::vector<Predictioin> df;
    dffg->process(sdf, df);
    
    return 0;
}