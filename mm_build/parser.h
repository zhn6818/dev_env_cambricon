/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#ifndef PARSER_H_
#define PARSER_H_
#include <dlfcn.h>
#include "mm_parser.h"
#include "common/logger.h"
#include "common/macros.h"
#include "build_param.h"
using namespace magicmind;

template <ModelKind Kind>
struct ParseTraits;
// Gcc 4.8.5 can not trait func type as static member variable, so use static member func instead.
template <>
struct ParseTraits<ModelKind::kCaffe> {
  typedef IParser<ModelKind::kCaffe, std::string, std::string> ParserT;
  typedef ParserT *(*Creator)();
  static Creator GetCreator() {
    return CreateIParser<ModelKind::kCaffe, std::string, std::string>;
  }
};

template <>
struct ParseTraits<ModelKind::kOnnx> {
  typedef IParser<ModelKind::kOnnx, std::string> ParserT;
  typedef ParserT *(*Creator)();
  static Creator GetCreator() {
    return CreateIParser<ModelKind::kOnnx, std::string>;
  }
};

template <>
struct ParseTraits<ModelKind::kPytorch> {
  typedef IParser<ModelKind::kPytorch, std::string> ParserT;
  typedef ParserT *(*Creator)();
  static Creator GetCreator() {
    return CreateIParser<ModelKind::kPytorch, std::string>;
  }
};

template <>
struct ParseTraits<ModelKind::kTensorflow> {
  typedef IParser<ModelKind::kTensorflow, std::string> ParserT;
  typedef ParserT *(*Creator)();
  static Creator GetCreator() {
    return CreateIParser<ModelKind::kTensorflow, std::string>;
  }
};


// Simple class wrapper pointer for model parser
// Ref to `mm_parser.h` for more details about `class IParser`
template <ModelKind Kind>
class ModelParser {
  using RawParserT = typename ParseTraits<Kind>::ParserT;

 public:
  ModelParser() = delete;
  explicit ModelParser(ParserParam<Kind> *param)
      : parser_raw_ptr_(ParseTraits<Kind>::GetCreator()()), param_(param) {
    CHECK_VALID(parser_raw_ptr_);
    if (HasValue(param->plugin())) {
      auto paths = Value(param->plugin());
      for (auto path : paths) {
        path         = AddLocalPathIfName(path);
        void *handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!handle) {
          SLOG(ERROR) << "Call dlopen() failed : " << dlerror();
          abort();
        }
        dlhandler_vec_.push_back(handle);
      }
    }
  }

  ~ModelParser() {
    for (auto handle : dlhandler_vec_) {
      dlclose(handle);
    }
    parser_raw_ptr_->Destroy();
  }

  void Parse(INetwork *network);

 private:
  RawParserT *parser_raw_ptr_;
  ParserParam<Kind> *param_;
  std::vector<void *> dlhandler_vec_;
};

#endif  // PARSER_H_
