#include <iostream>
#include "mm_network.h"
#include "mm_parser.h"
#include "mm_builder.h"
using namespace magicmind;
int main()
{
    std::cout << "hello world " << std::endl;
    auto net = CreateINetwork();
    IParser<ModelKind::kOnnx, std::string> *parser = CreateIParser<ModelKind::kOnnx, std::string>();
    parser->Parse(net, "/usr/local/neuware/samples/magicmind/mm_build/model/resnet18.onnx");
    IBuilderConfig *config = CreateIBuilderConfig();
    config->ParseFromFile("user_config_path/build_config.json");
    
    return 0;
}