#include "builder_config.h"

void BindCluster(std::stringstream *ss,
                 const std::vector<std::string> &mlu_arch,
                 const std::vector<std::vector<int>> &cluster_num,
                 size_t index) {
  *ss << "{\"";
  *ss << mlu_arch[index];
  *ss << "\":";
  if (cluster_num.size() > index) {
    *ss << "[";
    for (size_t j = 0; j < cluster_num[index].size(); ++j) {
      *ss << std::to_string(cluster_num[index][j]);
      if (j < cluster_num[index].size() - 1) {
        *ss << ",";
      }
    }
    *ss << "]}";
  } else {
    // support empty array
    *ss << "[]}";
  }
  return;
}

std::string GetChannelOppositeLayout(const std::string &in)
{
  static const std::unordered_map<std::string, std::string> layouts = {
      {"NCT", "NTC"}, {"NCHW", "NHWC"}, {"NCDHW", "NDHWC"}};
  for (auto e_ : layouts)
  {
    if (in == e_.first)
    {
      return e_.second;
    }
    else if (in == e_.second)
    {
      return e_.first;
    }
  }
  SLOG(ERROR) << "Unsupport layout convertion";
  abort();
}
IBuilderConfig *GetConfig(BuildParam *param)
{
    auto config_ptr_ = CreateIBuilderConfig();
    CHECK_VALID(config_ptr_);
    // arch
    CHECK_STATUS(config_ptr_->SetMLUArch(Value(param->mlu_arch())));
    // precision
    if (HasValue(param->precision()))
    {
        CHECK_STATUS(config_ptr_->ParseFromString("{\"precision_config\":{\"precision_mode\":\"" +
                                                  Value(param->precision()) + "\"}}"));
    }
    // cluster num
    auto cluster_num = Value(param->cluster_num());
    auto mlu_arch = Value(param->mlu_arch());
    if (cluster_num.size() > 0 && mlu_arch.size() > 0)
    {
        // set bitmap of visible cluster for each architecture
        // example: "archs": [{"mtp_372": [cluster_num_1, cluster_num_2, cluster_num_3]}]
        std::stringstream ss;
        ss << "{\"archs\": [";
        for (size_t i = 0; i < mlu_arch.size(); ++i)
        {
            BindCluster(&ss, mlu_arch, cluster_num, i);
            if (i < mlu_arch.size() - 1)
            {
                ss << ",";
            }
            else
            {
                ss << "]";
            }
        }
        ss << "}";
        CHECK_STATUS(config_ptr_->ParseFromString(ss.str()));
    }

    // means/vars
    if (HasValue(param->means()))
    {
        CHECK_VALID(HasValue(param->vars()));
        auto means = Value(param->means());
        auto vars = Value(param->vars());
        std::stringstream ss;
        ss << "{\"insert_bn_before_firstnode\":{";
        for (size_t i = 0; i < means.size(); ++i)
        {
            ss << "\"" << std::to_string(i) << "\":{\"mean\":";
            ss << means[i];
            ss << ",\"var\":";
            ss << vars[i];
            ss << "}";
            if (i < means.size() - 1)
            {
                ss << ",";
            }
            else
            {
                ss << "}";
            }
        }
        ss << "}";
        CHECK_STATUS(config_ptr_->ParseFromString(ss.str()));
    }
    // cross_compile
    if (HasValue(param->toolchain_path()))
    {
        CHECK_STATUS(config_ptr_->ParseFromString("{\"cross_compile_toolchain_path\":\"" +
                                                  Value(param->toolchain_path()) + "\"}"));
    }
    // layout in/out
    if (HasValue(param->input_layout()))
    {
        std::stringstream ss;
        auto v = Value(param->input_layout());
        ss << "{\"convert_input_layout\":{";
        for (size_t i = 0; i < v.size(); ++i)
        {
            ss << "\"" << std::to_string(i) << "\":{\"src\":\"";
            ss << GetChannelOppositeLayout(v[i]) << "\",";
            ss << "\"dst\":\"";
            ss << v[i];
            ss << "\"}";
            if (i < v.size() - 1)
            {
                ss << ",";
            }
            else
            {
                ss << "}";
            }
        }
        ss << "}";
        CHECK_STATUS(config_ptr_->ParseFromString(ss.str()));
    }
    if (HasValue(param->output_layout()))
    {
        std::stringstream ss;
        auto v = Value(param->output_layout());
        ss << "{\"convert_output_layout\":{";
        for (size_t i = 0; i < v.size(); ++i)
        {
            ss << "\"" << std::to_string(i) << "\":{\"src\":\"";
            ss << GetChannelOppositeLayout(v[i]) << "\",";
            ss << "\"dst\":\"";
            ss << v[i];
            ss << "\"}";
            if (i < v.size() - 1)
            {
                ss << ",";
            }
            else
            {
                ss << "}";
            }
        }
        ss << "}";
        CHECK_STATUS(config_ptr_->ParseFromString(ss.str()));
    }
    if (HasValue(param->dynamic_shape()))
    {
        if (!Value(param->dynamic_shape()))
        {
            CHECK_STATUS(config_ptr_->ParseFromString("{\"graph_shape_mutable\":false}"));
        }
    }
    // rest json
    if (HasValue(param->build_config()))
    {
        CHECK_STATUS(config_ptr_->ParseFromFile(Value(param->build_config())));
    }
    return config_ptr_;
}