/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#ifndef BUILD_PARAM_H_
#define BUILD_PARAM_H_
#include <vector>
#include <string>
#include "mm_common.h"
#include "common/param.h"
#include "common/logger.h"

using namespace magicmind;

inline std::vector<std::string> AlternativeLayouts() {
  return {"NCHW", "NHWC", "NCT", "NTC", "NCDHW", "NDHWC"};
}

inline std::vector<std::string> AlternativeDtypes() {
  return {"INT8", "INT16", "INT32", "UINT8", "UINT16", "UINT32",
          "HALF", "FLOAT", "BOOL",  "QINT8", "QINT16"};
}

inline QuantizationAlgorithm StringToAlgo(const std::string &algo) {
  if (algo == "linear") {
    return QuantizationAlgorithm::LINEAR_ALGORITHM;
  } else if (algo == "eqnm") {
    return QuantizationAlgorithm::EQNM_ALGORITHM;
  }
  return QuantizationAlgorithm::INVALID;
}

class BuildParam : public ArgListBase {
  DECLARE_ARG(precision, (std::string))
      ->SetDescription("Mix precision mode.")
      ->SetAlternative({"force_float32", "force_float16", "qint16_mixed_float16",
                        "qint8_mixed_float16", "qint16_mixed_float32", "qint8_mixed_float32"})
      ->SetDefault({});
  DECLARE_ARG(calibration, (bool))
      ->SetDescription(
          "To do calibration or not. Will use range [-1,1] and skip calibration if no file or "
          "range is set.")
      ->SetDefault({"false"});
  DECLARE_ARG(rpc_server, (std::string))
      ->SetDescription("Set remote address for calibration.")
      ->SetDefault({});
  DECLARE_ARG(calibration_algo, (std::string))
      ->SetDescription("Set quantization algorithm for calibration.")
      ->SetAlternative({"linear", "eqnm"})
      ->SetDefault({"linear"});
  DECLARE_ARG(file_list, (std::vector<std::string>))
      ->SetDescription(
          "Input file list path by order. For calibration only. MUST input with "
          "calibration_data_path.")
      ->SetDefault({});
  DECLARE_ARG(calibration_data_path, (std::string))
      ->SetDescription("Directory for calibration data. MUST input with file_list.")
      ->SetDefault({});
  DECLARE_ARG(random_calib_range, (std::vector<float>))
      ->SetDescription("Set random range for calibration. Will override path and filelist.")
      ->SetDefault({});
  DECLARE_ARG(batch_size, (std::vector<int>))
      ->SetDescription(
          "Input batchsize by order, will override all highest dimensions for all inputs and not "
          "affect unrank/scalar inputs.")
      ->SetDefault({});
  DECLARE_ARG(input_dims, (std::vector<std::vector<int>>))
      ->SetDescription("Input shapes by order. '_' represents scalar.")
      ->SetDefault({});
  DECLARE_ARG(input_layout, (std::vector<std::string>))
      ->SetDescription(
          "Convert input layouts from channel last to channel second (or the opposite) by order.")
      ->SetAlternative(AlternativeLayouts())
      ->SetDefault({});
  DECLARE_ARG(input_dtypes, (std::vector<std::string>))
      ->SetDescription("Input data types by order for inference (will not affect calibration).")
      ->SetAlternative(AlternativeDtypes())
      ->SetDefault({});
  DECLARE_ARG(dynamic_shape, (bool))
      ->SetDescription("To compile with dynamic shape or not.")
      ->SetDefault({"true"});
  DECLARE_ARG(vars, (std::vector<std::vector<float>>))
      ->SetDescription("Vars for inputs by order. MUST input with means.")
      ->SetDefault({});
  DECLARE_ARG(means, (std::vector<std::vector<float>>))
      ->SetDescription("Means for inputs by order. MUST input with vars.")
      ->SetDefault({});
  DECLARE_ARG(output_layout, (std::vector<std::string>))
      ->SetDescription(
          "Convert output layouts from channel last to channel second (or the opposite) by order.")
      ->SetAlternative(AlternativeLayouts())
      ->SetDefault({});
  DECLARE_ARG(output_dtypes, (std::vector<std::string>))
      ->SetDescription("Output data types by order.")
      ->SetAlternative(AlternativeDtypes())
      ->SetDefault({});
  DECLARE_ARG(mlu_arch, (std::vector<std::string>))
      ->SetDescription("Target arch for mlu dev. Unset means all.")
      ->SetAlternative({"mtp_372", "tp_322", "mtp_592"})
      ->SetDefault({});
  DECLARE_ARG(plugin, (std::vector<std::string>))
      ->SetDescription("Plugin library paths to link with.")
      ->SetDefault({});
  DECLARE_ARG(magicmind_model, (std::string))
      ->SetDescription("File path for output serialization model file.")
      ->SetDefault({"./model"});
  DECLARE_ARG(build_config, (std::string))
      ->SetDescription(
          "Additional json build config for build. Config json will override other arg params.")
      ->SetDefault({});
  DECLARE_ARG(toolchain_path, (std::string))
      ->SetDescription("Cross compile toolchain path for tp_322.")
      ->SetDefault({"/usr/local/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"});
  DECLARE_ARG(rgb2bgr, (bool))
      ->SetDescription("convert RGB to BGR for first layer's Conv/BatchNorm/Scale of network")
      ->SetDefault({"false"});
  DECLARE_ARG(cluster_num, (std::vector<std::vector<int>>))
      ->SetDescription("Allow users to flexibly specify the cluster")
      ->SetDefault({});
};

template <ModelKind Kind>
class ParserParam : public BuildParam {};

template <>
class ParserParam<ModelKind::kCaffe> : public BuildParam {
  DECLARE_ARG(prototxt, (std::string))->SetDescription("Prototxt file path for Caffe parser.");
  DECLARE_ARG(caffemodel, (std::string))->SetDescription("Caffemodel file path for Caffe parser.");
};

template <>
class ParserParam<ModelKind::kOnnx> : public BuildParam {
  DECLARE_ARG(onnx, (std::string))->SetDescription("Onnx file path for ONNX parser.");
};

template <>
class ParserParam<ModelKind::kPytorch> : public BuildParam {
  DECLARE_ARG(pytorch_pt, (std::string))
      ->SetDescription("PyTorch pt file path for PyTorch parser.");
  DECLARE_ARG(pt_input_dtypes, (std::vector<std::string>))
      ->SetDescription("Input data types by order for parsing PyTorch pt.")
      ->SetAlternative(AlternativeDtypes())
      ->SetDefault({"FLOAT"});
};

template <>
class ParserParam<ModelKind::kTensorflow> : public BuildParam {
  DECLARE_ARG(tf_pb, (std::string))->SetDescription("TensorFlow pb file for TensorFlow parser.");
  DECLARE_ARG(input_names, (std::vector<std::string>))
      ->SetDescription("Input names for TensorFlow parser.");
  DECLARE_ARG(output_names, (std::vector<std::string>))
      ->SetDescription("Output names for TensorFlow parser.");
};

#endif  // BUILD_PARAM_H_
