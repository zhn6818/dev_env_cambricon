#include "model_process.h"

ModelProcess::ModelProcess(int device)
{
    deviceId = device;
    cnrt_init();
}
ModelProcess::~ModelProcess()
{
    delete[] data_ptr;
    for (auto tensor : input_tensors)
    {
        cnrtFree(tensor->GetMutableData());
        tensor->Destroy();
    }
    for (auto tensor : output_tensors)
    {
        if (output_mlu_addr_ptr != nullptr)
        {
            cnrtFree(tensor->GetMutableData());
        }
        tensor->Destroy();
    }
    context->Destroy();
    engine->Destroy();
    model->Destroy();
}

void ModelProcess::cnrt_init()
{
    // 1. cnrt init
    std::cout << "Cnrt init..." << std::endl;
    MluDeviceGuard device_guard(deviceId);
    CHECK_CNRT(cnrtQueueCreate, &queue);
}
void ModelProcess::create_model(std::string path_model)
{
    // 2.create model
    std::cout << "Load model..." << std::endl;
    model = CreateIModel();
    CHECK_PTR(model);
    MM_CHECK(model->DeserializeFromFile(path_model.c_str()));
    PrintModelInfo(model);
}
void ModelProcess::create_engine()
{
    // 3. crete engine
    std::cout << "Create engine..." << std::endl;
    engine = model->CreateIEngine();
    CHECK_PTR(engine);
}

void ModelProcess::create_context()
{
    // 4. create context
    std::cout << "Create context..." << std::endl;
    magicmind::IModel::EngineConfig engine_config;
    engine_config.SetDeviceType("MLU");
    engine_config.SetConstDataInit(true);
    context = engine->CreateIContext();
    CHECK_PTR(context);
}

void ModelProcess::create_inout_tensor()
{
    // 5. crete input tensor and output tensor and memory alloc
    CHECK_MM(context->CreateInputTensors, &input_tensors);
    CHECK_MM(context->CreateOutputTensors, &output_tensors);
}

void ModelProcess::alloc_memory_input_mlu()
{
    // 6. memory alloc
    // input tensor mlu ptrs
    auto input_dim_vec = model->GetInputDimension(0).GetDims();
    if (input_dim_vec[0] == -1)
    {
        input_dim_vec[0] = 1;
    }
    magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
    input_tensors[0]->SetDimensions(input_dim);
    CNRT_CHECK(cnrtMalloc(&input_mlu_addr_ptr, input_tensors[0]->GetSize()));
    MM_CHECK(input_tensors[0]->SetData(input_mlu_addr_ptr));
}

void ModelProcess::alloc_memory_output_mlu()
{
    // 6. memory alloc
    // output tensor mlu ptrs
    if (magicmind::Status::OK() == context->InferOutputShape(input_tensors, output_tensors))
    {
        for (size_t output_id = 0; output_id < model->GetOutputNum(); ++output_id)
        {
            CNRT_CHECK(cnrtMalloc(&output_mlu_addr_ptr, output_tensors[output_id]->GetSize()));
            MM_CHECK(output_tensors[output_id]->SetData(output_mlu_addr_ptr));
        }
    }
    else
    {
        std::cout << "InferOutputShape failed" << std::endl;
    }
}

void ModelProcess::alloc_memory_output_cpu()
{
    // 6. memory alloc
    // output tensor cpu ptrs
    data_ptr = new float[output_tensors[0]->GetSize() / sizeof(output_tensors[0]->GetDataType())];
}