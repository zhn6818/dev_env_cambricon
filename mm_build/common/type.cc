/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include "common/type.h"
/*
 * Function to change strings to datatypes.
 */
std::vector<magicmind::DataType> ToDataType(const std::vector<std::string> &strings) {
  std::vector<magicmind::DataType> ret;
  for (auto e_ : strings) {
    ret.push_back(magicmind::TypeStringToEnum(e_));
  }
  return ret;
}
/*
 * Function to change vector ints to dims.
 */
std::vector<magicmind::Dims> ToDims(const std::vector<std::vector<int>> &shapes) {
  std::vector<magicmind::Dims> ret;
  for (auto e_ : shapes) {
    ret.push_back(magicmind::Dims({e_.begin(), e_.end()}));
  }
  return ret;
}
/*
 * Function to change strings to layouts.
 */
std::vector<magicmind::Layout> ToLayouts(const std::vector<std::string> &strings) {
  std::vector<magicmind::Layout> ret;
  for (auto e_ : strings) {
    ret.push_back(magicmind::LayoutStringToEnum(e_));
  }
  return ret;
}
