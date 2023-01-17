/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include "common/type.h"
#include "parser.h"

template <>
void ModelParser<ModelKind::kCaffe>::Parse(INetwork *network) {
  CHECK_STATUS(parser_raw_ptr_->Parse(network, Value(param_->caffemodel()).c_str(),
                                      Value(param_->prototxt()).c_str()));
}

template <>
void ModelParser<ModelKind::kOnnx>::Parse(INetwork *network) {
  CHECK_STATUS(parser_raw_ptr_->Parse(network, Value(param_->onnx()).c_str()));
}

template <>
void ModelParser<ModelKind::kPytorch>::Parse(INetwork *network) {
  std::vector<DataType> parse_dtypes = ToDataType(Value(param_->pt_input_dtypes()));
  CHECK_STATUS(parser_raw_ptr_->SetModelParam("pytorch-input-dtypes", parse_dtypes));
  CHECK_STATUS(parser_raw_ptr_->Parse(network, Value(param_->pytorch_pt()).c_str()));
}

template <>
void ModelParser<ModelKind::kTensorflow>::Parse(INetwork *network) {
  CHECK_STATUS(parser_raw_ptr_->SetModelParam("tf-model-type", "tf-graphdef-file"));
  CHECK_STATUS(parser_raw_ptr_->SetModelParam("tf-graphdef-inputs", Value(param_->input_names())));
  CHECK_STATUS(
      parser_raw_ptr_->SetModelParam("tf-graphdef-outputs", Value(param_->output_names())));
  CHECK_STATUS(parser_raw_ptr_->Parse(network, Value(param_->tf_pb()).c_str()));
}
