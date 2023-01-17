#include "postprocess.h"
using namespace std;


void Postprocess::detect_result(int detect_num, float *out, std::vector<std::vector<float>> det_results)
{
    vector<vector<float>> output;
    for(int i = 0 ; i < detect_num ; ++i) 
    {
        std::vector<float> result;
        float class_idx = *(out+7*i+1);
        float score = *(out+7*i+2);
        float xmin = *(out+7*i+3);
        float ymin = *(out+7*i+4);
        float xmax = *(out+7*i+5);
        float ymax = *(out+7*i+6);

        result.push_back(class_idx);
        result.push_back(score);
        result.push_back(xmin);
        result.push_back(ymin);
        result.push_back(xmax);
        result.push_back(ymax);
        output.push_back(result);
    }

};

void Postprocess::post_process(cv::Mat &img, int detect_num, float *out, std::vector<std::string> labels, int dst_h, int dst_w)
{

    //获取n个目标的类别、得分、框坐标
    vector<vector<float>> output;
    for(int i = 0 ; i < detect_num ; ++i) 
    {
        std::vector<float> result;
        float class_idx = *(out+7*i+1);
        float score = *(out+7*i+2);
        float xmin = *(out+7*i+3);
        float ymin = *(out+7*i+4);
        float xmax = *(out+7*i+5);
        float ymax = *(out+7*i+6);

        result.push_back(class_idx);
        result.push_back(score);
        result.push_back(xmin);
        result.push_back(ymin);
        result.push_back(xmax);
        result.push_back(ymax);
        output.push_back(result);
    }

    //处理n个目标的类别、得分、框坐标
    int src_h = img.rows;
    int src_w = img.cols;
    float post_dst_h = dst_h;
    float post_dst_w = dst_w;
    float ratio = std::min(float(post_dst_h)/float(src_h), float(post_dst_w)/float(src_w));
    float scale_w = ratio * src_w;
    float scale_h = ratio * src_h;

    for (int i = 0; i < detect_num; i++)
    {
        int detect_class = output[i][0];
        float score = output[i][1];
        float xmin = output[i][2];
        float ymin = output[i][3];
        float xmax = output[i][4];
        float ymax = output[i][5];

        xmin = std::max(float(0.0), std::min(xmin, post_dst_w));
        xmax = std::max(float(0.0), std::min(xmax, post_dst_w));
        ymin = std::max(float(0.0), std::min(ymin, post_dst_h));
        ymax = std::max(float(0.0), std::min(ymax, post_dst_h));
        xmin = (xmin - (post_dst_w - scale_w)/2)/ratio;
        ymin = (ymin - (post_dst_h - scale_h)/2)/ratio;
        xmax = (xmax - (post_dst_w - scale_w)/2)/ratio;
        ymax = (ymax - (post_dst_h - scale_h)/2)/ratio;
        xmin = std::max(0.0f, float(xmin));
        xmax = std::max(0.0f, float(xmax));
        ymin = std::max(0.0f, float(ymin));
        ymax = std::max(0.0f, float(ymax));
        
        std::cout << "detect_class:" << labels[detect_class] << ";"
                    << "score:" << score << ";"
                    << "xmin:" << xmin << ";"
                    << "ymin:" << ymin << ";"
                    << "xmax:" << xmax << ";"
                    << "ymax:" << ymax << ";"
                    << std::endl;
    } 

}

// void Postprocess::post_process(cv::Mat &img, int detect_num, std::vector<std::vector<float>> output, std::vector<std::string> labels)
// {

//     int src_h = img.rows;
//     int src_w = img.cols;
//     float post_dst_h = 640;
//     float post_dst_w = 640;
//     float ratio = std::min(float(post_dst_h)/float(src_h), float(post_dst_w)/float(src_w));
//     float scale_w = ratio * src_w;
//     float scale_h = ratio * src_h;

//     for (int i = 0; i < detect_num; i++)
//     {
//         int detect_class = output[i][0];
//         float score = output[i][1];
//         float xmin = output[i][2];
//         float ymin = output[i][3];
//         float xmax = output[i][4];
//         float ymax = output[i][5];

//         xmin = std::max(float(0.0), std::min(xmin, post_dst_w));
//         xmax = std::max(float(0.0), std::min(xmax, post_dst_w));
//         ymin = std::max(float(0.0), std::min(ymin, post_dst_h));
//         ymax = std::max(float(0.0), std::min(ymax, post_dst_h));
//         xmin = (xmin - (post_dst_w - scale_w)/2)/ratio;
//         ymin = (ymin - (post_dst_h - scale_h)/2)/ratio;
//         xmax = (xmax - (post_dst_w - scale_w)/2)/ratio;
//         ymax = (ymax - (post_dst_h - scale_h)/2)/ratio;
//         xmin = std::max(0.0f, float(xmin));
//         xmax = std::max(0.0f, float(xmax));
//         ymin = std::max(0.0f, float(ymin));
//         ymax = std::max(0.0f, float(ymax));
        
//         std::cout << "detect_class:" << labels[detect_class] << ";"
//                     << "score:" << score << ";"
//                     << "xmin:" << xmin << ";"
//                     << "ymin:" << ymin << ";"
//                     << "xmax:" << xmax << ";"
//                     << "ymax:" << ymax << ";"
//                     << std::endl;
//     } 

// }