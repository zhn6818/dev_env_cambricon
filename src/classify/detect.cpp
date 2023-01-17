#include "cnrt_virgo.h"
#include "../Util/utils.h"
#include "../ModelProcess/model_process.h"
#include "../preprocess/preprocess.h"
#include "../postprocess/postprocess.h"
#include <glog/logging.h>
#include <json/reader.h>
#include <json/value.h>
using namespace std;

namespace CNRT_VIRGO
{
    class DetectPrivate
    {
    private:
    std::string namesPath;
    std::string jsonpath;
    std::vector<std::string> labels;
    cv::Mat to_img;
    float shift;
    

    public:
        DetectPrivate(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath)
        {
            std::cout << "this is test" << std::endl;
            modelProcess = std::make_shared<ModelProcess>(deviceId);
            preProcess = std::make_shared<Preprocess>();
            postProcess = std::make_shared<Postprocess>();
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

            // shift = 0;
            // std::cout << jsonPath << std::endl;
            // if (jsonPath.size() <= 0)
            // {
            //     shift = 0;
            //     std::cout << "shift: " << shift << std::endl;
            // }
            // else
            // {
            //     Json::Value root;
            //     Json::Reader reader;
            //     std::ifstream is(jsonPath);
            //     if (!is.is_open())
            //     {
            //         // LOG(INFO)("file is not opened");
            //     }
            //     else
            //     {
            //         reader.parse(is, root);
            //         shift = root["shift"].asFloat();
            //         std::cout << "shift: " << shift << std::endl;
            //     }
            // }
            // fin.clear();
            // fin.close();

        }

        void process(std::vector<cv::Mat> &vecMat, std::vector<Predictioin> &results)
        {
            results.resize(0);
            size_t inputSize = modelProcess->GetInputSize();
            size_t sizeIn = sizeof(modelProcess->GetInputType());
            cv::Mat img = vecMat[0];
            
            preProcess->detectPreprocess(vecMat[0],to_img, modelProcess->GetModelHW().model_h, modelProcess->GetModelHW().model_w);
            modelProcess->copyinputdata(to_img.data, inputSize);
            modelProcess->enqueue();
            float *out = (float *)modelProcess->copyoutputdata();
            int detect_num = (int ) modelProcess->copyoutputdatanum();
            std::cout << detect_num << std::endl;
            postProcess->post_process(img, detect_num, out, labels, modelProcess->GetModelHW().model_h, modelProcess->GetModelHW().model_w);

            
        }

    private:
        std::shared_ptr<ModelProcess> modelProcess;
        std::shared_ptr<Preprocess> preProcess;
        std::shared_ptr<Postprocess> postProcess;

    };
    
    Detect::Detect(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath)
    {
        m_pHandle = std::make_shared<DetectPrivate>(model_path, name_Path, deviceId, jsonPath);
    }

    Detect::~Detect()
    {
    }

    void Detect::process(std::vector<cv::Mat> &vecMat, std::vector<Predictioin> &results)
    {
        m_pHandle->process(vecMat, results);
    }

    size_t Detect::GetBatch()
    {
        
    }
    size_t Detect::GetInputSize()
    {
    }
}
