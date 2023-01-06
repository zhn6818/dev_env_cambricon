/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Some function to change string/vector<int> to Dims/DataType/Layout
 *************************************************************************/
#ifndef TYPE_H_
#define TYPE_H_
#include <string>
#include <vector>
#include "mm_common.h"
#include "third_party/half/half.h"
using half = half_float::half;
/*!
 * @struct DataTypeToEnum
 * Converts Basic data type in C/C++ to ::DataType enumerate value.
 * DataTypeToEnum<T>::value is the DataType enumerate value for basic data type T.
 * e.g. DataTypeToEnum<float>::value is DataType::FLOAT32.
 */
template <class T>
struct DataTypeToEnum {};

// Template specialization for DataTypeToEnum, EnumToDataType and DataTypeToString.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                \
  template <>                                          \
  struct DataTypeToEnum<TYPE> {                        \
    static constexpr magicmind::DataType value = ENUM; \
  };

MATCH_TYPE_AND_ENUM(int8_t, magicmind::DataType::INT8);
MATCH_TYPE_AND_ENUM(int16_t, magicmind::DataType::INT16);
MATCH_TYPE_AND_ENUM(int32_t, magicmind::DataType::INT32);
MATCH_TYPE_AND_ENUM(uint8_t, magicmind::DataType::UINT8);
MATCH_TYPE_AND_ENUM(uint16_t, magicmind::DataType::UINT16);
MATCH_TYPE_AND_ENUM(uint32_t, magicmind::DataType::UINT32);
MATCH_TYPE_AND_ENUM(half, magicmind::DataType::FLOAT16);
MATCH_TYPE_AND_ENUM(float, magicmind::DataType::FLOAT32);
MATCH_TYPE_AND_ENUM(bool, magicmind::DataType::BOOL);
#undef MATCH_TYPE_AND_ENUM
/*
 * Function to change strings to datatypes.
 */
std::vector<magicmind::DataType> ToDataType(const std::vector<std::string> &strings);
/*
 * Function to change vector ints to dims.
 */
std::vector<magicmind::Dims> ToDims(const std::vector<std::vector<int>> &shapes);
/*
 * Function to change strings to layouts.
 */
std::vector<magicmind::Layout> ToLayouts(const std::vector<std::string> &strings);

#endif  // TYPE_H_
