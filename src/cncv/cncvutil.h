/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/
#ifndef CNCV_SAMPLES_COMMON_UTILS_H_
#define CNCV_SAMPLES_COMMON_UTILS_H_

#include <string>
#include <iostream>
#include "opencv2/opencv.hpp"

#include "cncv.h"

#define callCNCVFunc(cncvFunc, ...)                                                               \
    {                                                                                             \
        cncvStatus_t ret_code = (cncvFunc);                                                       \
        if (CNCV_STATUS_SUCCESS != ret_code)                                                      \
        {                                                                                         \
            std::cout << "call cncv function " << #cncvFunc << " failed. error code:" << ret_code \
                      << ". (" << std::endl;                                                      \
        }                                                                                         \
    }

#define callCNRTFunc(cnrtFunc, ...)                                                               \
    {                                                                                             \
        cnrtRet_t ret_code = (cnrtFunc);                                                          \
        if (CNRT_RET_SUCCESS != ret_code)                                                         \
        {                                                                                         \
            std::cout << "call cnrt function " << #cnrtFunc << " failed. error code:" << ret_code \
                      << ". (" << std::endl;                                                      \
        }                                                                                         \
    }

#define LOG(str)                                                                          \
    {                                                                                     \
        std::cout << __FILE__ << ':' << __LINE__ << " " << std::string(str) << std::endl; \
    }

#define safeCheck(check_mode, str, ...) \
    {                                   \
        if (!check_mode)                \
        {                               \
            LOG(str);                   \
        }                               \
    }

void printCNCVVersion();

bool getDeviceName(char *device_name);

uint32_t getPixFmtPlaneNum(cncvPixelFormat pixfmt);

uint32_t getPixFmtChannelNum(cncvPixelFormat pixfmt);

uint32_t getSizeOfDepth(cncvDepth_t depth);

uint64_t getImageDataSize(const cncvImageDescriptor desc);

uint64_t getMaxImageDataSize(const size_t batch_size, const cncvImageDescriptor *pdescs);

uint64_t getVariableBatchDataSize(const size_t batch_size, const cncvImageDescriptor *pdescs);

uint64_t getFixedBatchDataSize(const size_t batch_size, const cncvImageDescriptor desc);

void *mallocDevice(uint64_t data_size);

bool readDataFromFile(void *pData, const std::string file_name, const uint32_t expect_data_size);

void saveDstImage(void *pData,
                  const std::string image_name,
                  const cncvImageDescriptor desc,
                  int type);

#endif // CNCV_SAMPLES_COMMON_UTILS_H_
