#include <iostream>
#include "mm_network.h"
#include "mm_parser.h"
#include "mm_builder.h"
#include "build_param.h"
#include "builder_config.h"
#include "common/type.h"
#include "common/logger.h"
#include "common/macros.h"
using namespace magicmind;

int main(int argc, char *argv[])
{
    std::cout << "hello world " << std::endl;
    auto net = CreateINetwork();
    IParser<ModelKind::kOnnx, std::string> *parser = CreateIParser<ModelKind::kOnnx, std::string>();
    std::string modelath = "/usr/local/neuware/samples/magicmind/mm_build/model/resnet18.onnx";
    parser->Parse(net, modelath);

    // IBuilderConfig *config = CreateIBuilderConfig();
    // config->ParseFromFile("user_config_path/build_config.json");
    auto args = ArrangeArgs(argc, argv);
    auto param = new ParserParam<ModelKind::kOnnx>();
    param->ReadIn(args);
    SLOG(INFO) << "\n==================== Parameter Information\n"
               << param->DebugString() << "MagicMind: " << MM_VERSION;
    CHECK_VALID(ConfigNetwork(net, param));
    auto config = GetConfig(param);



    
    CHECK_VALID(config);
    // CHECK_VALID(BuildAndSerialize(net, config, param));

    auto builder = CreateIBuilder();
    // auto model_name = Value(param->magicmind_model());
    std::string model_name = "./mm_model/resnet18.model1";
    if (!builder)
    {
        SLOG(ERROR) << "CreateIBuilder failed.";
        return false;
    }
    size_t input_count = net->GetInputCount();
    size_t output_count = net->GetOutputCount();
    if (HasValue(param->input_dtypes()))
    {
        SLOG(INFO) << "Reset input dtypes.";
        auto types = ToDataType(Value(param->input_dtypes()));
        if (input_count != types.size())
        {
            SLOG(ERROR) << "Got " << input_count << " inputs from network, but " << types.size()
                        << " from param->";
            return false;
        }
        for (size_t i = 0; i < input_count; ++i)
        {
            auto tensor = net->GetInput(i);
            if (!tensor->SetDataType(types[i]).ok())
            {
                return false;
            }
        }
    }
    if (HasValue(param->output_dtypes()))
    {
        SLOG(INFO) << "Reset output dtypes.";
        auto types = ToDataType(Value(param->output_dtypes()));
        if (output_count != types.size())
        {
            SLOG(ERROR) << "Got " << output_count << " inputs from network, but " << types.size()
                        << " from param->";
            return false;
        }
        for (size_t i = 0; i < output_count; ++i)
        {
            auto tensor = net->GetOutput(i);
            if (!tensor->SetDataType(types[i]).ok())
            {
                return false;
            }
        }
    }

    auto model = builder->BuildModel("network", net, config);
    if (!model)
    {
        SLOG(ERROR) << "BuildModel failed";
        return false;
    }
    auto ret = model->SerializeToFile(model_name.c_str());
    if (!ret.ok())
    {
        SLOG(ERROR) << "Serialization failed with " << ret.ToString();
        return false;
    }
    auto input_dtypes = model->GetInputDataTypes();
    auto output_dtypes = model->GetOutputDataTypes();
    auto input_names = model->GetInputNames();
    auto output_names = model->GetOutputNames();
    for (int i = 0; i < model->GetInputNum(); i++)
    {
        auto input_shape = model->GetInputDimensions()[i];
        SLOG(INFO) << " model input[" << i << "] name is : " << input_names[i];
        SLOG(INFO) << " model input[" << i << "] shape is : " << input_shape;
        SLOG(INFO) << " model input[" << i
                   << "] dtype is : " << magicmind::TypeEnumToString(input_dtypes[i]);
    }
    for (int i = 0; i < model->GetOutputNum(); i++)
    {
        auto output_shape = model->GetOutputDimensions()[i];
        SLOG(INFO) << " model output[" << i << "] name is : " << output_names[i];
        SLOG(INFO) << " model output[" << i << "] shape is : " << output_shape;
        SLOG(INFO) << " model output[" << i
                   << "] dtype is : " << magicmind::TypeEnumToString(output_dtypes[i]);
    }
    builder->Destroy();
    model->Destroy();
    delete param;
    return 0;
}