#include "cnrt_virgo.h"
#include "../Util/utils.h"
#include "../ModelProcess/model_process.h"
#include "../preprocess/preprocess.h"
#include "../cncv/cncvPre.h"
#include <glog/logging.h>
#include <json/reader.h>
#include <json/value.h>
using namespace std;

namespace CNRT_VIRGO
{
    class ClassifyPrivate
    {
    private:
        std::string namesPath;
        std::string jsonpath;
        std::vector<std::string> labels;
        float shift;

    public:
        ClassifyPrivate(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath)
        {
            std::cout << "this is test" << std::endl;
            modelProcess = std::make_shared<ModelProcess>(deviceId);
            preProcess = std::make_shared<Preprocess>();
            cncvPreprocess = std::make_shared<CNCVpreprocess>(deviceId);
            modelProcess->create_model(model_path);
            modelProcess->create_engine();
            modelProcess->create_context();
            modelProcess->create_inout_tensor();

            modelProcess->alloc_memory_input_mlu();
            modelProcess->alloc_memory_output_mlu();
            modelProcess->alloc_memory_output_cpu();

            namesPath = name_Path;
            std::ifstream fin(namesPath, std::ios::in);
            char line[1024] = {0};
            std::string name = "";
            while (fin.getline(line, sizeof(line)))
            {
                std::stringstream word(line);
                word >> name;
                std::cout << "name: " << name << std::endl;
                labels.push_back(name);
            }

            shift = 0;
            std::cout << jsonPath << std::endl;
            if (jsonPath.size() <= 0)
            {
                shift = 0;
                std::cout << "shift: " << shift << std::endl;
            }
            else
            {
                Json::Value root;
                Json::Reader reader;
                std::ifstream is(jsonPath);
                if (!is.is_open())
                {
                    // LOG(INFO)("file is not opened");
                }
                else
                {
                    reader.parse(is, root);
                    shift = root["shift"].asFloat();
                    std::cout << "shift: " << shift << std::endl;
                }
            }
            fin.clear();
            fin.close();
        }

        size_t get_batch()
        {
            return modelProcess->GetBatch();
        }

        size_t get_input_size()
        {
            return modelProcess->GetInputSize();
        }
        void Mat2ChannelLast(cv::Mat &src, float *p_input)
        {
            int CHANNEL = 3;
            assert(!src.empty());

            for (int i = 0; i < src.rows; i++)
            {
                for (int j = 0; j < src.cols; j++)
                {
                    for (int c = 0; c < CHANNEL; c++)
                    {
                        p_input[i * src.cols * CHANNEL + j * CHANNEL + c] = (src.ptr<float>(i)[j * CHANNEL + c]) / 1.0;
                    }
                }
            }
        }
        void process(std::vector<cv::Mat> &vecMat, std::vector<Predictioin> &result)
        {
            result.resize(0);
            size_t inputSize = modelProcess->GetInputSize();
            size_t sizeIn = sizeof(modelProcess->GetInputType());

            // float *data = new float[inputSize];
            // preProcess->classPreprocess(vecMat[0], data, modelProcess->GetModelHW().model_h, modelProcess->GetModelHW().model_w);
            // for(int i = 0; i < 490; i++)
            // {
            //     std::cout << " " << *(data + i);
            // }
            // std::cout << std::endl;

            // float *data22 = new float[inputSize];
            // cncvPreprocess->classPreprocess(vecMat[0], data22, modelProcess->GetModelHW().model_h, modelProcess->GetModelHW().model_w);

            // cncvPreprocess->cncvsplit();

            float *data = new float[inputSize];
            cncvPreprocess->cncvresizestd(vecMat[0], data, modelProcess->GetModelHW().model_h, modelProcess->GetModelHW().model_w);

            // for(int i = 0; i < 490; i++)
            // {
            //     std::cout << " " << (float)*(data + i);
            // }
            // std::cout << std::endl;

            // float *data = new float[inputSize];
            // cv::Mat img = cv::imread("./test/resize_00.png");
            // img.convertTo(img, CV_32FC3, 1 / 255.0);
            // Mat2ChannelLast(img, data);
            // std::cout << std::endl;


            modelProcess->copyinputdata(data, inputSize);
            modelProcess->enqueue();
            float *out = (float *)modelProcess->copyoutputdata();
            size_t outsize = modelProcess->GetOutputSize();
            size_t sizeOut = sizeof(modelProcess->GetOutputType());

            for (int i = 0; i < outsize / sizeOut; i++)
            {
                std::cout << " " << *(out + i);
            }
            std::cout << std::endl;

            float maxValue = *max_element(out, out + outsize / sizeOut);
            int maxPosition = max_element(out, out + outsize / sizeOut) - out;
            result.push_back(std::make_pair(labels[maxPosition], preProcess->prob_sigmoid(maxValue)));
        }

    private:
        std::shared_ptr<ModelProcess> modelProcess;
        std::shared_ptr<Preprocess> preProcess;
        std::shared_ptr<CNCVpreprocess> cncvPreprocess;
        ;
    };

    Classify::Classify(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath)
    {
        m_pHandlerClassifyPrivate = std::make_shared<ClassifyPrivate>(model_path, name_Path, deviceId, jsonPath);
    }
    Classify::~Classify()
    {
    }
    void Classify::process(std::vector<cv::Mat> &vecMat, std::vector<Predictioin> &result)
    {
        m_pHandlerClassifyPrivate->process(vecMat, result);
    }
    size_t Classify::GetBatch()
    {
        return m_pHandlerClassifyPrivate->get_batch();
    }
    size_t Classify::GetInputSize()
    {
        return m_pHandlerClassifyPrivate->get_input_size();
    }
}
