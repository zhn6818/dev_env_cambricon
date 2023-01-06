#pragma once

#include <vector>
#include <string>
#include "mm_builder.h"
#include "mm_calibrator.h"
#include "common/calib_data.h"
#include "build_param.h"
// #include "mm_build/build_param.h"
// #include "mm_build/parser.h"

using namespace magicmind;

IBuilderConfig *GetConfig(BuildParam *param);