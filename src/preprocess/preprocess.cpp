#include "preprocess.h"
using namespace std;






void Preprocess::Mat2ChannelFirst(cv::Mat &src, float* p_input) 
{
    int CHANNEL = 3;
    assert(!src.empty());
    for(int c = 0; c < CHANNEL; c++)
    {
        for(int i = 0; i < src.rows; i++)
        {
            for(int j = 0; j < src.cols; j++)
            {
                p_input[c * src.rows * src.cols + i * src.cols + j] = src.ptr<float>(i)[j * CHANNEL + c];
            }
        }
    }
}

void Preprocess::classPreprocess(cv::Mat &img, float* data, int h, int w)
{
    cv::resize(img, img, cv::Size(h, w), 0, 0, cv::INTER_CUBIC);

            
    img.convertTo(img, CV_32FC3, 1 / 255.0);
    Mat2ChannelFirst(img, data);
}

void Preprocess::detectPreprocess(cv::Mat &img, cv::Mat &to_img, int h, int w)
{
    int src_h = img.rows;
    int src_w = img.cols;
    int dst_h = h;
    int dst_w = w;
    float ratio = std::min(float(dst_h)/float(src_h), float(dst_w)/float(src_w));
    int unpad_h = std::floor(src_h * ratio);
    int unpad_w = std::floor(src_w * ratio);
    if(ratio !=1){
        int interpolation;
        if(ratio < 1){
            interpolation = cv::INTER_AREA;
        }else{
            interpolation = cv::INTER_LINEAR;
        }
        cv::resize(img, img, cv::Size(unpad_w, unpad_h), interpolation);
    }

    int pad_t = std::floor((dst_h - unpad_h)/2);
    int pad_b = dst_h - unpad_h - pad_t;
    int pad_l = std::floor((dst_w - unpad_w)/2);
    int pad_r = dst_w - unpad_w - pad_l;

    cv::copyMakeBorder(img, img, pad_t, pad_b, pad_l, pad_r, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    to_img = img;
}


float  Preprocess::prob_sigmoid(float x)
{
    return (1 / (1 + exp(-x)));

}
