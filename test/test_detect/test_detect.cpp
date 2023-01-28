#include <iostream>
#include "cnrt_virgo.h"
#include <dirent.h>
using namespace CNRT_VIRGO;

//yolov7 detect 3个输出
int main()
{
    std::shared_ptr<Detect> dffg = std::make_shared<Detect>("/data1/qwy/code/magicmind_cloud-master2/buildin/cv/detection/yolov7_pytorch/data/models/yolov7_pytorch_model_qint8_mixed_float16_true", 
                                                            "/data1/qwy/code/magicmind_cloud-master2/buildin/cv/utils/coco.names", 0);
    
    
    // *********************************************检测一张图************************************************
    std::string image_path = "/data1/qwy/dataset/coco/val2017/000000000632.jpg";
    cv::Mat img = cv::imread(image_path);
    std::vector<cv::Mat> sdf;
    sdf.push_back(img);
    std::vector<std::vector<DetectedObject>> df;
    dffg->process(sdf, df);
    std::cout << "process success" << std::endl;

    for (int i = 0; i < df[0].size(); i++)
    {
        std::cout << "image_path= " << image_path << "; "
                  << "labels=" << df[0][i].object_class <<"; "
                  << "score=" << df[0][i].prob << "; "
                  << "bounding_box=" << df[0][i].bounding_box << std::endl;
        
    }

    //************************************************检测n张图**************************************************

    // std::vector<std::string> image_paths;
    // std::string image_path = "/data1/qwy/dataset/coco/val2017_part";
    // auto dir = opendir(image_path.c_str());
 
    // if ((dir) != NULL)
    // {
    //     struct dirent *entry;
    //     entry = readdir(dir);
    //     int count = 0;
    //     while (entry)
    //     {
    //         auto temp = image_path + "/" + entry->d_name;
    //         if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
    //         {
    //             entry = readdir(dir);
    //             continue;
    //         }
    //         count = count + 1;
    //         image_paths.push_back(temp);
    //         entry = readdir(dir);
    //     }
    // }
    // std::cout << image_paths.size() << std::endl;


    // size_t image_num = image_paths.size();
    
    // for (int i = 0; i < image_num; i++)
    // {
    //     cv::Mat img = cv::imread(image_paths[i]);
    //     std::vector<cv::Mat> sdf;
    //     sdf.push_back(img);
    //     std::vector<std::vector<DetectedObject>> df;
    //     dffg->process(sdf, df);
    //     std::cout << "process success" << std::endl;

    //     for (int o = 0; o < df[0].size(); o++)
    //     {
    //         std::cout << "image_path=" << image_paths[i] << "; " 
    //                 << "labels=" << df[0][o].object_class << "; "
    //                 << "score=" << df[0][o].prob << "; "
    //                 << "bounding_box=" << df[0][o].bounding_box << std::endl;
    //     }
    // }
    
    // return 0;
}



//yolov7 detect 2个输出，labels和score，通
// int main()
// {
//     std::shared_ptr<Detect> dffg = std::make_shared<Detect>("/data1/qwy/code/magicmind_cloud-master2/buildin/cv/detection/yolov7_pytorch/data/models/yolov7_pytorch_model_qint8_mixed_float16_true", 
//                                                             "/data1/qwy/code/magicmind_cloud-master2/buildin/cv/utils/coco.names", 0);
//     // std::shared_ptr<Classify> dffg = std::make_shared<Classify>("/usr/local/neuware/samples/magicmind/mm_build/model/resnet18.model", 
//     //                                                             "/data1/qwy/labels/commonworker.names", 
//     //                                                             0);
    
//     // *********************batch == 1***********************
//     cv::Mat img = cv::imread("/data1/qwy/dataset/coco/test/000000000139.jpg");
//     std::vector<cv::Mat> sdf;
//     sdf.push_back(img);
//     std::vector<Predictioin> df;
//     dffg->process(sdf, df);
//     std::cout << "process success" << std::endl;

//     for (int i = 0; i < df.size(); i++)
//     {
//         std::cout << "labels=" << df[i].first <<"; "
//                   << "score=" << df[i].second << std::endl;
//     }
    
//     return 0;
// }