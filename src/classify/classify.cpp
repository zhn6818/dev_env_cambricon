#include "cnrt_virgo.h"
#include "../Util/utils.h"

namespace ASCEND_VIRGO
{
    class ClassifyPrivate
    {
    public:
        ClassifyPrivate(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath)
        {
            std::cout << "this is test" << std::endl;
            std::cout << "Cnrt init..." << std::endl;
            MluDeviceGuard device_guard(deviceId);
            cnrtQueue_t queue;
            CHECK_CNRT(cnrtQueueCreate, &queue);

            // 2.create model
            std::cout << "Load model..." << std::endl;
            auto model = CreateIModel();
            CHECK_PTR(model);
            MM_CHECK(model->DeserializeFromFile(model_path.c_str()));
            PrintModelInfo(model);
        }

    private:
        int a;
    };

    Classify::Classify(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath)
    {
        m_pHandlerClassifyPrivate = std::make_shared<ClassifyPrivate>(model_path, name_Path, deviceId, jsonPath);
    }
    Classify::~Classify()
    {
    }
    size_t Classify::GetBatch()
    {
    }
    size_t Classify::GetInputSize()
    {
    }
}
