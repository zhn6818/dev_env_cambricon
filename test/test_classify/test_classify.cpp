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

    // *********************读取一张图***********************
    // std::string image_path = "/data1/qwy/test.png";
    // cv::Mat img = cv::imread(image_path);
    // std::vector<cv::Mat> sdf;
    // sdf.push_back(img);
    // std::vector<Predictioin> df;
    // dffg->process(sdf, df);

    // for (int i = 0; i < df.size(); i++)
    // {
    //     std::cout << "image_path= " << image_path << "; "
    //               << "labels=" << df[0].first <<"; "
    //               << "score=" << df[0].second << "; "
    //               << std::endl;

    // }

    // return 0;

    //********************************读取n张图*****************************
    std::vector<std::string> image_paths;
    std::string image_path = "/data1/qwy/dataset/rail";
    auto dir = opendir(image_path.c_str());

    if ((dir) != NULL)
    {
        struct dirent *entry;
        entry = readdir(dir);
        int count = 0;
        while (entry)
        {
            auto temp = image_path + "/" + entry->d_name;
            if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            {
                entry = readdir(dir);
                continue;
            }
            count = count + 1;
            image_paths.push_back(temp);
            entry = readdir(dir);
        }
    }
    std::cout << image_paths.size() << std::endl;

    size_t image_num = image_paths.size();

    for (int i = 0; i < image_num; i++)
    {
        std::cout << "img name: " << image_paths[i] << std::endl;
        cv::Mat img = cv::imread(image_paths[i]);
        std::vector<cv::Mat> sdf;
        sdf.push_back(img);
        std::vector<Predictioin> df;
        dffg->process(sdf, df);

        for (int o = 0; o < df.size(); o++)
        {
            std::cout << "image_path= " << image_paths[i] << "; "
                      << "labels=" << df[0].first << "; "
                      << "score=" << df[0].second << "; "
                      << std::endl;
        }
    }

    return 0;
}