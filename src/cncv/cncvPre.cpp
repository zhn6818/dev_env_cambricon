#include "cncvPre.h"

CNCVpreprocess::CNCVpreprocess(int device)
{
    device_id = device;
    initial();
}

void CNCVpreprocess::initial()
{

    callCNRTFunc(cnrtSetDevice(device_id));
    callCNRTFunc(cnrtQueueCreate(&queue));
    callCNCVFunc(cncvCreate(&handle));
    callCNCVFunc(cncvSetQueue(handle, queue));
}

void CNCVpreprocess::printMat(uint8_t *mat, size_t length)
{
    std::cout << "this is cpu out:   ";
    for (int i = 0; i < length; i++)
    {
        std::cout << (int)mat[i] << " ";
    }
    std::cout << std::endl;
}

void CNCVpreprocess::printMat(float *mat, size_t length)
{
    std::cout << "this is cpu out:   ";
    for (int i = 0; i < length; i++)
    {
        std::cout << (float)mat[i] << " ";
    }
    std::cout << std::endl;
}

void CNCVpreprocess::release()
{
    free(img_cpu_buffer);
    free(cpu_src_ptrs);
    free(cpu_dst_ptrs);
    cnrtFree(workspace);
    cnrtFree(mlu_src_datas);
    cnrtFree(mlu_dst_datas);
    cnrtFree(mlu_src_ptrs);
    cnrtFree(mlu_dst_ptrs);
    if (handle)
    {
        cncvDestroy(handle);
        handle = nullptr;
    }
    if (queue)
    {
        cnrtQueueDestroy(queue);
        queue = nullptr;
    }
}

void CNCVpreprocess::classPreprocess(cv::Mat &src_mat, float *data, int w, int h)
{
    printMat(src_mat.data, 20);
    const uint32_t batch_size = 1;
    const cncvPixelFormat src_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst_fmt = src_fmt;
    const cncvDepth_t src_dtype = CNCV_DEPTH_8U;
    const cncvDepth_t dst_dtype = CNCV_DEPTH_32F;
    const cncvColorSpace color_space = CNCV_COLOR_SPACE_INVALID;
    uint32_t in_width = src_mat.cols;
    uint32_t in_height = src_mat.rows;
    uint32_t num_src_ptrs = getPixFmtPlaneNum(src_fmt) * batch_size;
    uint32_t num_dst_ptrs = getPixFmtPlaneNum(dst_fmt) * batch_size;

    cncvImageDescriptor src_desc;
    src_desc.width = in_width;
    src_desc.height = in_height;
    src_desc.depth = src_dtype;
    src_desc.pixel_fmt = src_fmt;
    src_desc.color_space = color_space;
    src_desc.stride[0] = 3 * in_width * getSizeOfDepth(src_desc.depth);

    cncvImageDescriptor dst_desc;
    dst_desc.width = in_width;
    dst_desc.height = in_height;
    dst_desc.depth = dst_dtype;
    dst_desc.pixel_fmt = dst_fmt;
    dst_desc.color_space = color_space;
    dst_desc.stride[0] = 3 * in_width * getSizeOfDepth(dst_desc.depth);
    float mean[3] = {0, 0, 0};
    float std[3] = {255.0, 255.0, 255.0};

    // prepare memory: malloc input&output data buffer, temp cpu buffer, and
    // workspace buffer
    uint64_t total_src_data_size = getFixedBatchDataSize(batch_size, src_desc);
    uint64_t total_dst_data_size = getFixedBatchDataSize(batch_size, dst_desc);
    uint64_t max_image_size =
        MAX(getImageDataSize(src_desc), getImageDataSize(dst_desc));
    size_t min_workspace_size = 0;
    size_t channel_num = 3;
    callCNCVFunc(cncvGetMeanStdWorkspaceSize(channel_num, &min_workspace_size));

    cpu_src_ptrs = (void **)malloc(num_src_ptrs * sizeof(void *));
    cpu_dst_ptrs = (void **)malloc(num_dst_ptrs * sizeof(void *));
    img_cpu_buffer = malloc(max_image_size);
    mlu_src_ptrs = (void **)mallocDevice(num_src_ptrs * sizeof(void *));
    mlu_dst_ptrs = (void **)mallocDevice(num_dst_ptrs * sizeof(void *));
    mlu_src_datas = mallocDevice(total_src_data_size);
    mlu_dst_datas = mallocDevice(total_dst_data_size);
    workspace = mallocDevice(min_workspace_size);
    safeCheck(cpu_src_ptrs && cpu_dst_ptrs && mlu_src_ptrs && mlu_dst_ptrs &&
                  mlu_src_datas && mlu_dst_datas && workspace,
              "some temp memory malloc failed.");

    // 2. transform data from host to device
    uint64_t src_data_offset = 0;
    uint64_t dst_data_offset = 0;
    for (uint32_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy((uint8_t *)mlu_src_datas + src_data_offset,
                                src_mat.data,
                                src_desc.stride[0] * src_desc.height,
                                CNRT_MEM_TRANS_DIR_HOST2DEV));

        cpu_src_ptrs[i] = (uint8_t *)mlu_src_datas + src_data_offset;
        cpu_dst_ptrs[i] = (uint8_t *)mlu_dst_datas + dst_data_offset;

        src_data_offset += src_desc.stride[0] * src_desc.height;
        dst_data_offset += dst_desc.stride[0] * dst_desc.height;
    }

    callCNRTFunc(cnrtMemcpy(mlu_src_ptrs,
                            cpu_src_ptrs,
                            num_src_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    callCNRTFunc(cnrtMemcpy(mlu_dst_ptrs,
                            cpu_dst_ptrs,
                            num_dst_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 3. call cncv operator and sync to get result
    callCNCVFunc(cncvMeanStd(handle,
                             batch_size,
                             src_desc,
                             mlu_src_ptrs,
                             mean,
                             std,
                             dst_desc,
                             mlu_dst_ptrs,
                             min_workspace_size,
                             workspace));
    callCNRTFunc(cnrtQueueSync(queue));

    for (size_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy(img_cpu_buffer,
                                cpu_dst_ptrs[i],
                                dst_desc.stride[0] * dst_desc.height,
                                CNRT_MEM_TRANS_DIR_DEV2HOST));
        std::string target_name = "./test/mean_std_output_" + std::to_string(i) + ".jpg";
        saveDstImage(img_cpu_buffer, target_name, dst_desc, CV_32FC3);
        printMat((float *)img_cpu_buffer, 20);
    }
    std::cout << std::endl;
}

void CNCVpreprocess::cncvsplit()
{
    uchar a[12] = {4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3};
    cv::Mat src_mat = cv::Mat(2, 2, CV_8UC3, a);
    const uint32_t batch_size = 1;
    const cncvPixelFormat src_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst_fmt = CNCV_PIX_FMT_GRAY;
    const cncvDepth_t src_dtype = CNCV_DEPTH_8U;
    const cncvDepth_t dst_dtype = CNCV_DEPTH_8U;
    const cncvColorSpace color_space = CNCV_COLOR_SPACE_INVALID;

    uint32_t in_width = src_mat.cols;
    uint32_t in_height = src_mat.rows;
    uint32_t num_src_ptrs = getPixFmtPlaneNum(src_fmt) * batch_size;
    uint32_t num_dst_ptrs = getPixFmtPlaneNum(dst_fmt) * batch_size * 3;

    cncvImageDescriptor src_desc;
    src_desc.width = in_width;
    src_desc.height = in_height;
    src_desc.depth = src_dtype;
    src_desc.pixel_fmt = src_fmt;
    src_desc.color_space = color_space;
    src_desc.stride[0] = 3 * in_width * getSizeOfDepth(src_desc.depth);

    cncvImageDescriptor dst_desc;
    dst_desc.width = in_width;
    dst_desc.height = in_height;
    dst_desc.depth = dst_dtype;
    dst_desc.pixel_fmt = dst_fmt;
    dst_desc.color_space = color_space;
    dst_desc.stride[0] = in_width * getSizeOfDepth(dst_desc.depth);
    dst_desc.stride[1] = dst_desc.stride[0];
    dst_desc.stride[2] = dst_desc.stride[0];

    cncvRect src_roi;
    src_roi.x = 0;
    src_roi.y = 0;
    src_roi.h = in_height;
    src_roi.w = in_width;

    uint64_t total_src_data_size = getFixedBatchDataSize(batch_size, src_desc);
    uint64_t total_dst_data_size = getFixedBatchDataSize(batch_size * 3, dst_desc);
    cpu_src_ptrs = (void **)malloc(num_src_ptrs * sizeof(void *));
    cpu_dst_ptrs = (void **)malloc(num_dst_ptrs * sizeof(void *));
    mlu_src_ptrs = (void **)mallocDevice(num_src_ptrs * sizeof(void *));
    mlu_dst_ptrs = (void **)mallocDevice(num_dst_ptrs * sizeof(void *));
    mlu_src_datas = mallocDevice(total_src_data_size);
    mlu_dst_datas = mallocDevice(total_dst_data_size);

    uint64_t src_data_offset = 0;
    for (uint32_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy((uint8_t *)mlu_src_datas + src_data_offset,
                                src_mat.data,
                                src_desc.stride[0] * src_desc.height,
                                CNRT_MEM_TRANS_DIR_HOST2DEV));
        cpu_src_ptrs[i] = (uint8_t *)mlu_src_datas + src_data_offset;
        src_data_offset += src_desc.stride[0] * src_desc.height;
    }
    uint64_t dst_data_offset = 0;
    for (uint32_t i = 0; i < batch_size * 3; i++)
    {
        cpu_dst_ptrs[i] = (uint8_t *)mlu_dst_datas + dst_data_offset;
        dst_data_offset += dst_desc.stride[0] * dst_desc.height;
    }
    callCNRTFunc(cnrtMemcpy(mlu_src_ptrs,
                            cpu_src_ptrs,
                            num_src_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));

    callCNRTFunc(cnrtMemcpy(mlu_dst_ptrs,
                            cpu_dst_ptrs,
                            num_dst_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    callCNCVFunc(cncvSplit_ROI(
        handle,
        batch_size,
        src_desc,
        src_roi,
        mlu_src_ptrs,
        dst_desc,
        mlu_dst_ptrs));

    uint64_t max_image_size = MAX(getImageDataSize(src_desc), getImageDataSize(dst_desc));
    img_cpu_buffer = malloc(max_image_size);
    dst_data_offset = 0;
    for (size_t i = 0; i < batch_size * 3; i++)
    {
        callCNRTFunc(cnrtMemcpy(img_cpu_buffer + dst_data_offset,
                                cpu_dst_ptrs[i],
                                dst_desc.stride[0] * dst_desc.height,
                                CNRT_MEM_TRANS_DIR_DEV2HOST));
        dst_data_offset += dst_desc.stride[0] * dst_desc.height;
    }
    for (int i = 0; i < max_image_size; i++)
    {
        std::cout << (int)static_cast<uchar *>(img_cpu_buffer)[i] << std::endl;
    }

    std::cout << std::endl;
}

CNCVpreprocess::~CNCVpreprocess()
{
    release();
}
