#include "../Util/utils.h"
using namespace magicmind;

class ModelProcess
{

public:
    ModelProcess(int device);

    ~ModelProcess();

    void cnrt_init();

    void create_model(std::string path_model);

    void create_engine();

    void create_context();

    void create_inout_tensor();

    void alloc_memory_input_mlu();

    void alloc_memory_output_mlu();

    void alloc_memory_output_cpu();

    size_t GetInputSize();

    size_t GetOutputSize();

    magicmind::DataType GetInputType();
    magicmind::DataType GetOutputType();

    void copyinputdata(void *pInput, size_t pInputSize);

    void enqueue();

    void* copyoutputdata();


private:
    int deviceId;
    cnrtQueue_t queue;
    magicmind::IModel *model;
    magicmind::IEngine *engine;
    magicmind::IContext *context;
    std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;

    void *input_mlu_addr_ptr;
    void *output_mlu_addr_ptr = nullptr;
    float *data_ptr = nullptr;
};