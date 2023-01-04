#include "cnrt_virgo.h"
#include "../Util/utils.h"
#include "../ModelProcess/model_process.h"

namespace ASCEND_VIRGO
{
    class ClassifyPrivate
    {
    public:
        ClassifyPrivate(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath)
        {
            std::cout << "this is test" << std::endl;
            modelProcess = std::make_shared<ModelProcess>(deviceId);
            modelProcess->create_model(model_path);
            modelProcess->create_engine();
            modelProcess->create_context();
            modelProcess->create_inout_tensor();

            modelProcess->alloc_memory_input_mlu();
            modelProcess->alloc_memory_output_mlu();
            modelProcess->alloc_memory_output_cpu();
        }

        void process(std::vector<cv::Mat> &vecMat, std::vector<Predictioin> &result)
        {
            result.resize(0);
            size_t inputSize = modelProcess->GetInputSize();
            size_t sizeIn = sizeof(modelProcess->GetInputType());
            float *data = new float[inputSize];
            for (int i = 0; i < inputSize / sizeIn; i++)
            {
                *(data + i) = 1;
            }
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
        }

    private:
        std::shared_ptr<ModelProcess> modelProcess;
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
    }
    size_t Classify::GetInputSize()
    {
    }
}
