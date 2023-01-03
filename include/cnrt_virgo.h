#pragma once

#include <iostream>
#include <memory>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <ostream>
#include <sstream>
#include "cnrt.h"
#include <mm_runtime.h>

using namespace magicmind;

namespace ASCEND_VIRGO
{
    class ClassifyPrivate;
    class Classify
    {
    public:
        Classify(const std::string &model_path, const std::string &name_Path, size_t deviceId, std::string jsonPath = "");
        ~Classify();

        size_t GetBatch();
        size_t GetInputSize();

    private:
        std::shared_ptr<ClassifyPrivate> m_pHandlerClassifyPrivate;
        // std::shared_ptr<ClassifyDvpp> m_pHandlerClassifyDvpp;
    };
};