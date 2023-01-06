/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Derived implements for CalibDataInterface.
 *************************************************************************/
#include "common/calib_data.h"
#include "common/data.h"
#include "common/logger.h"
#include "common/macros.h"
#include "common/type.h"

SampleCalibData::SampleCalibData(const magicmind::Dims &shape,
                                 const magicmind::DataType &data_type,
                                 int max_samples,
                                 const std::vector<std::string> &data_paths)
    : SampleCalibData(std::vector<magicmind::Dims>({shape}), data_type, max_samples, data_paths) {
}

SampleCalibData::SampleCalibData(const std::vector<magicmind::Dims> &shapes,
                                 const magicmind::DataType &data_type,
                                 int max_samples,
                                 const std::vector<std::string> &data_paths)
    : shapes_(shapes), data_type_(data_type), max_samples_(max_samples), data_paths_(data_paths) {
  Init();
}

SampleCalibData::SampleCalibData(const magicmind::Dims &shape, const magicmind::DataType &data_type)
    : SampleCalibData(shape, data_type, shape.GetDimValue(0), 0, 0) {
  SLOG(INFO) << "Use zeros as calibration init data.";
}

SampleCalibData::SampleCalibData(const magicmind::Dims &shape,
                                 const magicmind::DataType &data_type,
                                 int max_samples,
                                 float min,
                                 float max)
    : shapes_(std::vector<magicmind::Dims>({shape})),
      data_type_(data_type),
      max_samples_(max_samples),
      min_(min),
      max_(max),
      use_rand_(true) {
  Init();
}

void SampleCalibData::Init() {
  CHECK_LE(0, shapes_.size());
  size_t max_size = 0;
  for (auto shape : shapes_) {
    auto batch_size = shape.GetDimValue(0);
    CHECK_LE(0, batch_size);
    batch_sizes_.push_back(batch_size);
    size_t size = DataTypeSize(data_type_) * shape.GetElementCount();
    CHECK_LE(0, size);
    max_size = max_size >= size ? max_size : size;
  }
  buffer_ = malloc(max_size);
}

magicmind::Dims SampleCalibData::GetShape() const {
  if (current_shape_idx_ == -1) {
    return magicmind::Dims();
  }
  return shapes_[current_shape_idx_];
}

magicmind::DataType SampleCalibData::GetDataType() const {
  return data_type_;
}

void SampleCalibData::MoveShapeNext() {
  if (shapes_.size() > 1) {
    current_shape_idx_ = (current_shape_idx_ + 1) == int(batch_sizes_.size()) ? -1 : current_shape_idx_ + 1;
  } else {
    current_shape_idx_ = 0;
  }
}

magicmind::Status SampleCalibData::Next() {
  MoveShapeNext();
  if (shapes_.size() == 1) {
    if ((current_shape_idx_ < 0)
        || (current_sample_ + batch_sizes_[current_shape_idx_] > max_samples_)) {
      return magicmind::Status(magicmind::error::Code::OUT_OF_RANGE,
                               "Sample number is bigger than max sample number");
    }
  } else {
    if ((current_shape_idx_ < 0)
        || (current_sample_ + 1 > max_samples_)) {
      return magicmind::Status(magicmind::error::Code::OUT_OF_RANGE,
                               "Sample number is bigger than max sample number");
    }
  }
  if (use_rand_) {
    if (!FillFromRand()) {
      return magicmind::Status(magicmind::error::Code::INTERNAL, "Bad init for calib data");
    }
  } else {
    if (!FillFromFile()) {
      return magicmind::Status(magicmind::error::Code::INTERNAL, "Bad init for calib data");
    }
  }
  return magicmind::Status::OK();
}

magicmind::Status SampleCalibData::Reset() {
  current_sample_ = 0;
  current_shape_idx_ = 0;
  return magicmind::Status::OK();
}

void *SampleCalibData::GetSample() {
  return buffer_;
}

bool SampleCalibData::FillFromFile() {
  bool ret = true;
  if (shapes_.size() == 1) {
    // Fill static
    // several img fill one data
    std::vector<std::string> paths(data_paths_.begin() + current_sample_,
                                   data_paths_.begin() + current_sample_ + batch_sizes_.front());
    int index = 0;
    size_t single_size = shapes_.front().GetElementCount() * DataTypeSize(data_type_) / batch_sizes_.front();
    for (auto p_ : paths) {
      SLOG(INFO) << "Calibration: reading file " << p_;
      ret = ReadDataFromFile(p_, ((char *)buffer_ + index * single_size), single_size) && ret;
      ++current_sample_;
      ++index;
    }
  } else {
    // fill dynamic
    // one img for one data
    SLOG(INFO) << "Calibration: reading file " << data_paths_[current_sample_];
    ret = ReadDataFromFile(data_paths_[current_sample_], buffer_, shapes_[current_shape_idx_].GetElementCount() * DataTypeSize(data_type_));
    ++current_sample_;
  }
  return ret;
}

bool SampleCalibData::FillFromRand() {
  auto count = shapes_[current_shape_idx_].GetElementCount();
#define CASE(type)                                                              \
  case DataTypeToEnum<type>::value: {                                           \
    auto vec = GenRand<type>(count, static_cast<type>(min_), \
                             static_cast<type>(max_), 0);                       \
    std::copy(vec.begin(), vec.end(), static_cast<type *>(buffer_));            \
    current_sample_ += batch_sizes_[current_shape_idx_];                        \
    return true;                                                                \
  }
  switch (data_type_) {
    CASE(int8_t);
    CASE(int16_t);
    CASE(int32_t);
    CASE(uint8_t);
    CASE(uint16_t);
    CASE(uint32_t);
    CASE(float);
    // rand float data then cast to half
    case magicmind::DataType::FLOAT16: {
      auto vec = GenRand<float>(count, min_, max_, 0);
      auto ret = NormalCast(buffer_, data_type_, vec.data(), magicmind::DataType::FLOAT32,
                            count, false);
      current_sample_ += batch_sizes_[current_shape_idx_];
      return (ret.ok());
    }
    default:
      SLOG(ERROR) << "Unsupport datatype for calib_data " << TypeEnumToString(data_type_);
      return false;
  }
#undef CASE
}
