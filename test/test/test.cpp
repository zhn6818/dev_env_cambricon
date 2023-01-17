#include <iostream>
#include "cnrt_virgo.h"
#include <dirent.h>
using namespace CNRT_VIRGO;
int main()
{
    // std::shared_ptr<Classify> dffg = std::make_shared<Classify>("/data1/qwy/code/magicmind_cloud-master/buildin/cv/classification/resnet50_onnx/data/models/resnet50_onnx_model_force_float32_true_1", "", 0);
    std::shared_ptr<Classify> dffg = std::make_shared<Classify>("/usr/local/neuware/samples/magicmind/mm_build/model/resnet18.model", 
                                                                "/data1/qwy/labels/commonworker.names", 
                                                                0);

    // //*****************batch == n****************************
    // std::vector<std::string> image_paths;
    // int batch_size = 10;
    // std::string image_path = "/data1/qwy/dataset/rail";
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
    //         // std::cout << image_path + "/" + entry->d_name << std::endl;
    //         count = count + 1;
    //         image_paths.push_back(temp);
    //         entry = readdir(dir);
    //     }

    //     size_t pad_num = batch_size - image_paths.size() % batch_size;
    //     if (pad_num != batch_size) {
    //         std::cout << "There are " << image_paths.size() << " images in total, add " << pad_num
    //             << " more images to make the number of images is an integral multiple of batchsize[" << batch_size << "].";
    //         while (pad_num--)
    //             image_paths.emplace_back(*image_paths.rbegin());
    // }
    //     // std::cout << count << std::endl;
    //     // std::cout << image_paths.size() << std::endl;
    // }

    // size_t image_num = image_paths.size();
    // std::vector<cv::Mat> sdf;
    // for (int i = 0; i < image_num; i++)
    // {
    
    //     cv::Mat img = cv::imread(image_paths[i]);
    //     sdf.push_back(img);
        
    // }
    
    // std::vector<Predictioin> df;
    // dffg->process(sdf, df);
    // return 0;
    
    // *********************batch == 1***********************
    cv::Mat img = cv::imread("/data1/qwy/test.png");
    std::vector<cv::Mat> sdf;
    sdf.push_back(img);
    std::vector<Predictioin> df;
    dffg->process(sdf, df);
    
    return 0;
}