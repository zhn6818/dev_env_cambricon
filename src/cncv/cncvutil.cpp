/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#include "cncvutil.h"
#include "cnrt.h"
#include "cn_api.h"

void printCNCVVersion()
{
    int major, minor, patch = 0;
    cncvGetLibVersion(&major, &minor, &patch);

    LOG("cncv Version: " + std::to_string(major) + "." + std::to_string(minor) +
        "." + std::to_string(patch));
}

bool getDeviceName(char *device_name)
{
    int dev_ordinal;
    if (CNRT_RET_SUCCESS != cnrtGetDevice(&dev_ordinal))
    {
        LOG("call cnrtGetDevice failed.");
        return false;
    }

    CNdev mlu_dev;
    if (CN_SUCCESS != cnDeviceGet(&mlu_dev, dev_ordinal))
    {
        LOG("call cnDeviceGet failed.");
        return false;
    }

    if (CN_SUCCESS != cnDeviceGetName(device_name, 256, mlu_dev))
    {
        LOG("call cnDeviceGetName failed.");
        return false;
    }

    return true;
}

uint32_t getPixFmtPlaneNum(cncvPixelFormat pixfmt)
{
    switch (pixfmt)
    {
    case CNCV_PIX_FMT_NV12:
    case CNCV_PIX_FMT_NV21:
        return 2;
    case CNCV_PIX_FMT_I420:
    case CNCV_PIX_FMT_YV12:
        return 3;
    case CNCV_PIX_FMT_YUYV:
    case CNCV_PIX_FMT_UYVY:
    case CNCV_PIX_FMT_YVYU:
    case CNCV_PIX_FMT_VYUY:
    case CNCV_PIX_FMT_RGB:
    case CNCV_PIX_FMT_BGR:
    case CNCV_PIX_FMT_ARGB:
    case CNCV_PIX_FMT_ABGR:
    case CNCV_PIX_FMT_BGRA:
    case CNCV_PIX_FMT_RGBA:
    case CNCV_PIX_FMT_GRAY:
    case CNCV_PIX_FMT_LAB:
        return 1;
    default:
        return 0;
    }

    return 0;
}

uint32_t getPixFmtChannelNum(cncvPixelFormat pixfmt)
{
    switch (pixfmt)
    {
    case CNCV_PIX_FMT_GRAY:
        return 1;
    case CNCV_PIX_FMT_NV12:
    case CNCV_PIX_FMT_NV21:
    case CNCV_PIX_FMT_I420:
    case CNCV_PIX_FMT_YV12:
    case CNCV_PIX_FMT_YUYV:
    case CNCV_PIX_FMT_UYVY:
    case CNCV_PIX_FMT_YVYU:
    case CNCV_PIX_FMT_VYUY:
    case CNCV_PIX_FMT_RGB:
    case CNCV_PIX_FMT_BGR:
    case CNCV_PIX_FMT_LAB:
    case CNCV_PIX_FMT_HSV:
    case CNCV_PIX_FMT_HSV_FULL:
        return 3;
    case CNCV_PIX_FMT_ARGB:
    case CNCV_PIX_FMT_ABGR:
    case CNCV_PIX_FMT_BGRA:
    case CNCV_PIX_FMT_RGBA:
        return 4;
    default:
        LOG("Unsupported Pixel Format, Size = 0 by default.");
        return 0;
    }

    return 0;
}

uint32_t getSizeOfDepth(cncvDepth_t depth)
{
    switch (depth)
    {
    case CNCV_DEPTH_8U:
    case CNCV_DEPTH_8S:
        return 1;
    case CNCV_DEPTH_16U:
    case CNCV_DEPTH_16S:
    case CNCV_DEPTH_16F:
        return 2;
    case CNCV_DEPTH_32U:
    case CNCV_DEPTH_32S:
    case CNCV_DEPTH_32F:
        return 4;
    default:
        LOG("Unsupported Depth, Size = 0 by default.");
        return 0;
    }

    return 0;
}

uint64_t getImageDataSize(const cncvImageDescriptor desc)
{
    uint64_t image_data_size = 0;

    for (uint32_t i = 0; i < getPixFmtPlaneNum(desc.pixel_fmt); i++)
    {
        image_data_size += desc.height * desc.stride[i];
    }

    return image_data_size;
}

uint64_t getVariableBatchDataSize(const size_t batch_size,
                                  const cncvImageDescriptor *pDesc)
{
    if (!pDesc)
        return 0;

    uint64_t total_data_size = 0;
    for (uint32_t i = 0; i < batch_size; i++)
    {
        total_data_size += getImageDataSize(pDesc[i]);
    }

    return total_data_size;
}

uint64_t getFixedBatchDataSize(const size_t batch_size,
                               const cncvImageDescriptor desc)
{
    uint64_t total_data_size = getImageDataSize(desc);
    total_data_size *= batch_size;
    return total_data_size;
}

uint64_t getMaxImageDataSize(const size_t batch_size,
                             const cncvImageDescriptor *pdescs)
{
    if (nullptr == pdescs)
        return 0;
    uint64_t max_data_size = 0;

    for (uint32_t i = 0; i < batch_size; i++)
    {
        uint64_t image_size = getImageDataSize(pdescs[i]);
        max_data_size = image_size > max_data_size ? image_size : max_data_size;
    }

    return max_data_size;
}

void *mallocDevice(uint64_t data_size)
{
    void *pData = nullptr;
    cnrtRet_t ret_code = cnrtMalloc(reinterpret_cast<void **>(&pData), data_size);
    if (CNRT_RET_SUCCESS != ret_code)
    {
        std::cout << "call cnrtMalloc failed. error code:" << ret_code << ". ("
                  << std::endl;
        return nullptr;
    }

    return pData;
}

bool readDataFromFile(void *pData,
                      const std::string file_name,
                      const uint32_t expect_data_size)
{
    if (!pData)
        return false;
    FILE *pFile = fopen(file_name.c_str(), "rb");
    if (fread(pData, 1, expect_data_size, pFile) != expect_data_size)
    {
        fclose(pFile);
        return false;
    }

    fclose(pFile);
    return true;
}
void saveDstImage(void *img_cpu_buffer,
                  const std::string target_name,
                  const cncvImageDescriptor dst_desc,
                  int type)
{
    if (nullptr == img_cpu_buffer || target_name.size() <= 0)
    {
        LOG("saveDstImage failed, img_cpu_buffer is nullptr or target_name is "
            "empty.");
        return;
    }

    cv::Mat bgr_data(dst_desc.width, dst_desc.height, type, img_cpu_buffer);
    cv::imwrite(target_name, bgr_data);
    LOG("save one dst image, target image name: " + target_name);
}
