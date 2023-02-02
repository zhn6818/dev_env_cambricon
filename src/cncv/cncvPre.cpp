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
    std::cout << "this is cpu out:   " << std::endl;
    for (int i = 0; i < length; i++)
    {
        std::cout << (int)mat[i] << " ";
    }
    std::cout << std::endl;
}

void CNCVpreprocess::printMat(float *mat, size_t length)
{
    std::cout << "this is cpu out:   " << std::endl;
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
    free(cpu_dst2_ptrs);
    cnrtFree(workspace);
    cnrtFree(mlu_src_datas);
    cnrtFree(mlu_dst_datas);
    // cnrtFree(mlu_dst2_datas);
    cnrtFree(mlu_src_ptrs);
    cnrtFree(mlu_dst_ptrs);
    cnrtFree(mlu_dst2_ptrs);
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
    float a[12] = {1, 1, 1, 2, 2, 2, 3, 4, 3, 4, 4, 4};
    cv::Mat src_mat = cv::Mat(2, 2, CV_32FC3, a);
    const uint32_t batch_size = 1;
    const cncvPixelFormat src_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst_fmt = CNCV_PIX_FMT_GRAY;
    const cncvDepth_t src_dtype = CNCV_DEPTH_32F;
    const cncvDepth_t dst_dtype = CNCV_DEPTH_32F;
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
        callCNRTFunc(cnrtMemcpy((float *)mlu_src_datas + src_data_offset,
                                src_mat.data,
                                src_desc.stride[0] * src_desc.height,
                                CNRT_MEM_TRANS_DIR_HOST2DEV));
        cpu_src_ptrs[i] = (float *)mlu_src_datas + src_data_offset;
        src_data_offset += src_desc.stride[0] * src_desc.height / sizeof(float);
    }
    uint64_t dst_data_offset = 0;
    for (uint32_t i = 0; i < batch_size * 3; i++)
    {
        cpu_dst_ptrs[i] = (float *)mlu_dst_datas + dst_data_offset;
        dst_data_offset += dst_desc.stride[0] * dst_desc.height / sizeof(float);
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
    callCNRTFunc(cnrtQueueSync(queue));

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
    for (int i = 0; i < max_image_size / sizeof(float); i++)
    {
        std::cout << (float)static_cast<float *>(img_cpu_buffer)[i] << std::endl;
    }

    std::cout << std::endl;
}

void CNCVpreprocess::cncvresize()
{
    uchar a[12] = {10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40};
    cv::Mat src_mat = cv::Mat(2, 2, CV_8UC3, a);
    const uint32_t batch_size = 1;
    const cncvPixelFormat src_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst_fmt = CNCV_PIX_FMT_BGR;
    const cncvDepth_t src_dtype = CNCV_DEPTH_8U;
    const cncvDepth_t dst_dtype = CNCV_DEPTH_8U;
    const cncvColorSpace color_space = CNCV_COLOR_SPACE_INVALID;

    uint32_t in_width = src_mat.cols;
    uint32_t in_height = src_mat.rows;
    uint32_t out_width = src_mat.cols * 2;
    uint32_t out_height = src_mat.rows * 2;
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
    dst_desc.width = out_width;
    dst_desc.height = out_height;
    dst_desc.depth = dst_dtype;
    dst_desc.pixel_fmt = dst_fmt;
    dst_desc.color_space = color_space;
    dst_desc.stride[0] = 3 * out_width * getSizeOfDepth(dst_desc.depth);

    cncvImageDescriptor *src_descs = nullptr, *dst_descs = nullptr;
    src_descs = (cncvImageDescriptor *)malloc(batch_size * sizeof(cncvImageDescriptor));
    dst_descs = (cncvImageDescriptor *)malloc(batch_size * sizeof(cncvImageDescriptor));
    cncvRect *src_rois = nullptr, *dst_rois = nullptr;
    src_rois = (cncvRect *)malloc(batch_size * sizeof(cncvRect));
    dst_rois = (cncvRect *)malloc(batch_size * sizeof(cncvRect));
    // img descriptor & rectRois
    for (int i = 0; i < batch_size; i++)
    {
        src_descs[i] = src_desc;
        dst_descs[i] = dst_desc;
        src_rois[i].x = 0;
        src_rois[i].y = 0;
        src_rois[i].w = src_descs[i].width;
        src_rois[i].h = src_descs[i].height;
        dst_rois[i].x = 0;
        dst_rois[i].y = 0;
        dst_rois[i].w = dst_descs[i].width;
        dst_rois[i].h = dst_descs[i].height;
    }
    uint64_t total_src_data_size = getVariableBatchDataSize(batch_size, src_descs);
    uint64_t total_dst_data_size = getVariableBatchDataSize(batch_size, dst_descs);

    // worksize
    size_t workspace_resize;
    callCNCVFunc(cncvGetResizeWorkspaceSize(batch_size,
                                            src_descs,
                                            src_rois,
                                            dst_descs,
                                            dst_rois,
                                            CNCV_INTER_BILINEAR,
                                            &workspace_resize));
    workspace = mallocDevice(workspace_resize);
    cpu_src_ptrs = (void **)malloc(num_src_ptrs * sizeof(void *));
    cpu_dst_ptrs = (void **)malloc(num_dst_ptrs * sizeof(void *));
    mlu_src_ptrs = (void **)mallocDevice(num_src_ptrs * sizeof(void *));
    mlu_dst_ptrs = (void **)mallocDevice(num_dst_ptrs * sizeof(void *));
    mlu_src_datas = mallocDevice(total_src_data_size);
    mlu_dst_datas = mallocDevice(total_dst_data_size);

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
    callCNCVFunc(
        cncvResize_AdvancedROI(handle,
                               batch_size,
                               src_descs,
                               src_rois,
                               mlu_src_ptrs,
                               dst_descs,
                               dst_rois,
                               mlu_dst_ptrs,
                               workspace_resize,
                               (uint8_t *)workspace,
                               CNCV_INTER_BILINEAR));
    callCNRTFunc(cnrtQueueSync(queue));

    for (size_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy(img_cpu_buffer,
                                cpu_dst_ptrs[i],
                                dst_desc.stride[0] * dst_desc.height,
                                CNRT_MEM_TRANS_DIR_DEV2HOST));

        printMat((uchar *)img_cpu_buffer, dst_desc.stride[0] * dst_desc.height);
    }
    std::cout << std::endl;
}

void CNCVpreprocess::cncvresizestdsplit(cv::Mat &src_mat, float *data, int w, int h)
{
    // uchar a[12] = {10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40};
    // cv::Mat src_mat = cv::Mat(2, 2, CV_8UC3, a);
    // for (int i = 0; i < 490; i++)
    // {
    //     std::cout << (int)src_mat.data[i] << " ";
    // }
    std::cout << std::endl;
    const uint32_t batch_size = 1;
    const cncvPixelFormat src_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst2_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst3_fmt = CNCV_PIX_FMT_GRAY;
    const cncvDepth_t src_dtype = CNCV_DEPTH_8U;
    const cncvDepth_t dst_dtype = CNCV_DEPTH_8U;
    const cncvDepth_t dst2_dtype = CNCV_DEPTH_32F;
    const cncvDepth_t dst3_dtype = CNCV_DEPTH_32F;
    const cncvColorSpace color_space = CNCV_COLOR_SPACE_INVALID;

    uint32_t in_width = src_mat.cols;
    uint32_t in_height = src_mat.rows;
    uint32_t out_width = w;
    uint32_t out_height = h;
    uint32_t num_src_ptrs = getPixFmtPlaneNum(src_fmt) * batch_size;
    uint32_t num_dst_ptrs = getPixFmtPlaneNum(dst_fmt) * batch_size;
    uint32_t num_dst2_ptrs = getPixFmtPlaneNum(dst2_fmt) * batch_size;
    uint32_t num_dst3_ptrs = getPixFmtChannelNum(dst3_fmt) * batch_size * 3;

    cncvImageDescriptor src_desc;
    src_desc.width = in_width;
    src_desc.height = in_height;
    src_desc.depth = src_dtype;
    src_desc.pixel_fmt = src_fmt;
    src_desc.color_space = color_space;
    src_desc.stride[0] = 3 * in_width * getSizeOfDepth(src_desc.depth);

    cncvImageDescriptor dst_desc;
    dst_desc.width = out_width;
    dst_desc.height = out_height;
    dst_desc.depth = dst_dtype;
    dst_desc.pixel_fmt = dst_fmt;
    dst_desc.color_space = color_space;
    dst_desc.stride[0] = 3 * out_width * getSizeOfDepth(dst_desc.depth);

    cncvImageDescriptor dst2_desc;
    dst2_desc.width = out_width;
    dst2_desc.height = out_height;
    dst2_desc.depth = dst2_dtype;
    dst2_desc.pixel_fmt = dst2_fmt;
    dst2_desc.color_space = color_space;
    dst2_desc.stride[0] = 3 * out_width * getSizeOfDepth(dst2_desc.depth);

    cncvImageDescriptor dst3_desc;
    dst3_desc.width = out_width;
    dst3_desc.height = out_height;
    dst3_desc.depth = dst3_dtype;
    dst3_desc.pixel_fmt = dst3_fmt;
    dst3_desc.color_space = color_space;
    dst3_desc.stride[0] = out_width * getSizeOfDepth(dst3_desc.depth);
    dst3_desc.stride[1] = dst3_desc.stride[0];
    dst3_desc.stride[2] = dst3_desc.stride[0];

    cncvImageDescriptor *src_descs = nullptr, *dst_descs = nullptr;
    src_descs = (cncvImageDescriptor *)malloc(batch_size * sizeof(cncvImageDescriptor));
    dst_descs = (cncvImageDescriptor *)malloc(batch_size * sizeof(cncvImageDescriptor));
    cncvRect *src_rois = nullptr, *dst_rois = nullptr;
    src_rois = (cncvRect *)malloc(batch_size * sizeof(cncvRect));
    dst_rois = (cncvRect *)malloc(batch_size * sizeof(cncvRect));
    // img descriptor & rectRois
    for (int i = 0; i < batch_size; i++)
    {
        src_descs[i] = src_desc;
        dst_descs[i] = dst_desc;
        src_rois[i].x = 0;
        src_rois[i].y = 0;
        src_rois[i].w = src_descs[i].width;
        src_rois[i].h = src_descs[i].height;
        dst_rois[i].x = 0;
        dst_rois[i].y = 0;
        dst_rois[i].w = dst_descs[i].width;
        dst_rois[i].h = dst_descs[i].height;
    }

    cncvRect dst2_roi;
    dst2_roi.x = 0;
    dst2_roi.y = 0;
    dst2_roi.h = out_height;
    dst2_roi.w = out_width;

    uint64_t total_src_data_size = getVariableBatchDataSize(batch_size, src_descs);
    uint64_t total_dst_data_size = getVariableBatchDataSize(batch_size, dst_descs);
    uint64_t total_dst2_data_size = getFixedBatchDataSize(batch_size, dst2_desc);
    uint64_t total_dst3_data_size = getFixedBatchDataSize(batch_size * 3, dst3_desc);

    uint64_t max_image_size = MAX(getMaxImageDataSize(batch_size, src_descs),
                                  getImageDataSize(dst3_desc));
    img_cpu_buffer = malloc(max_image_size);
    // worksize
    size_t workspace_resize, min_workspace_size, total_workspace_size;
    callCNCVFunc(cncvGetResizeWorkspaceSize(batch_size,
                                            src_descs,
                                            src_rois,
                                            dst_descs,
                                            dst_rois,
                                            CNCV_INTER_BILINEAR,
                                            &workspace_resize));

    size_t channel_num = 3;
    callCNCVFunc(cncvGetMeanStdWorkspaceSize(channel_num, &min_workspace_size));
    total_workspace_size = workspace_resize + min_workspace_size;
    workspace = mallocDevice(total_workspace_size);

    cpu_src_ptrs = (void **)malloc(num_src_ptrs * sizeof(void *));
    cpu_dst_ptrs = (void **)malloc(num_dst_ptrs * sizeof(void *));
    cpu_dst2_ptrs = (void **)malloc(num_dst2_ptrs * sizeof(void *));
    cpu_dst3_ptrs = (void **)malloc(num_dst3_ptrs * sizeof(void *));

    mlu_src_ptrs = (void **)mallocDevice(num_src_ptrs * sizeof(void *));
    mlu_dst_ptrs = (void **)mallocDevice(num_dst_ptrs * sizeof(void *));
    mlu_dst2_ptrs = (void **)mallocDevice(num_dst2_ptrs * sizeof(void *));
    mlu_dst3_ptrs = (void **)mallocDevice(num_dst3_ptrs * sizeof(void *));

    mlu_src_datas = mallocDevice(total_src_data_size);
    mlu_dst_datas = mallocDevice(total_dst_data_size + total_dst2_data_size + total_dst3_data_size);
    callCNRTFunc(cnrtMemset((uint8_t *)mlu_dst_datas + total_dst_data_size, 0, total_dst2_data_size));
    callCNRTFunc(cnrtMemset((uint8_t *)mlu_dst_datas + total_dst_data_size + total_dst2_data_size, 0, total_dst3_data_size));

    uint64_t src_data_offset = 0;
    uint64_t dst_data_offset = 0;
    uint64_t dst2_data_offset = 0;
    uint64_t dst3_data_offset = 0;

    for (uint32_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy((uint8_t *)mlu_src_datas + src_data_offset,
                                src_mat.data,
                                src_desc.stride[0] * src_desc.height,
                                CNRT_MEM_TRANS_DIR_HOST2DEV));

        cpu_src_ptrs[i] = (uint8_t *)mlu_src_datas + src_data_offset;
        cpu_dst_ptrs[i] = (uint8_t *)mlu_dst_datas + dst_data_offset;
        cpu_dst2_ptrs[i] = (uint8_t *)mlu_dst_datas + dst2_data_offset + total_dst_data_size;

        for (uint32_t j = 0; j < 3; j++)
        {
            cpu_dst3_ptrs[3 * i + j] = (uint8_t *)mlu_dst_datas + total_dst_data_size + total_dst2_data_size + dst3_data_offset;
            dst3_data_offset += dst3_desc.stride[0] * dst3_desc.height;
        }

        src_data_offset += src_desc.stride[0] * src_desc.height;
        dst_data_offset += dst_desc.stride[0] * dst_desc.height;
        dst2_data_offset += dst2_desc.stride[0] * dst2_desc.height;
    }

    callCNRTFunc(cnrtMemcpy(mlu_src_ptrs,
                            cpu_src_ptrs,
                            num_src_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    callCNRTFunc(cnrtMemcpy(mlu_dst_ptrs,
                            cpu_dst_ptrs,
                            num_dst_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    callCNRTFunc(cnrtMemcpy(mlu_dst2_ptrs,
                            cpu_dst2_ptrs,
                            num_dst2_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));

    callCNRTFunc(cnrtMemcpy(mlu_dst3_ptrs,
                            cpu_dst3_ptrs,
                            num_dst3_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    callCNCVFunc(
        cncvResize_AdvancedROI(handle,
                               batch_size,
                               src_descs,
                               src_rois,
                               mlu_src_ptrs,
                               dst_descs,
                               dst_rois,
                               mlu_dst_ptrs,
                               workspace_resize,
                               (uint8_t *)workspace,
                               CNCV_INTER_BILINEAR));
    float mean[3] = {0, 0, 0};
    float std[3] = {255.0, 255.0, 255.0};
    callCNCVFunc(cncvMeanStd(handle,
                             batch_size,
                             dst_desc,
                             mlu_dst_ptrs,
                             mean,
                             std,
                             dst2_desc,
                             mlu_dst2_ptrs,
                             min_workspace_size,
                             (uint8_t *)workspace + workspace_resize));
    callCNCVFunc(
        cncvSplit_ROI(
            handle,
            batch_size,
            dst2_desc,
            dst2_roi,
            mlu_dst2_ptrs,
            dst3_desc,
            mlu_dst3_ptrs));
    callCNRTFunc(cnrtQueueSync(queue));

    for (size_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy(img_cpu_buffer,
                                cpu_dst_ptrs[i],
                                dst_desc.stride[0] * dst_desc.height,
                                CNRT_MEM_TRANS_DIR_DEV2HOST));
        std::string target_name = "./test/resize_" + std::to_string(i) + ".jpg";
        saveDstImage(img_cpu_buffer, target_name, dst_desc, CV_8UC3);
        printMat((uchar *)img_cpu_buffer, 490);
    }
    for (size_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy(img_cpu_buffer,
                                cpu_dst2_ptrs[i],
                                dst2_desc.stride[0] * dst2_desc.height,
                                CNRT_MEM_TRANS_DIR_DEV2HOST));

        printMat((float *)img_cpu_buffer, 490);
    }
    uint32_t offset = 0;
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            callCNRTFunc(cnrtMemcpy(img_cpu_buffer + offset,
                                    cpu_dst3_ptrs[i],
                                    dst_desc.stride[0] * dst_desc.height,
                                    CNRT_MEM_TRANS_DIR_DEV2HOST));
            offset += dst_desc.stride[0] * dst_desc.height;
        }

        printMat((float *)img_cpu_buffer, 490);
    }
    std::cout << std::endl;
}

void CNCVpreprocess::cncvresizestd(cv::Mat &src_mat, void *data, int w, int h)
{
    const uint32_t batch_size = 1;
    const cncvPixelFormat src_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst_fmt = CNCV_PIX_FMT_BGR;
    const cncvPixelFormat dst2_fmt = CNCV_PIX_FMT_BGR;

    const cncvDepth_t src_dtype = CNCV_DEPTH_8U;
    const cncvDepth_t dst_dtype = CNCV_DEPTH_8U;
    const cncvDepth_t dst2_dtype = CNCV_DEPTH_32F;

    const cncvColorSpace color_space = CNCV_COLOR_SPACE_INVALID;

    uint32_t in_width = src_mat.cols;
    uint32_t in_height = src_mat.rows;
    uint32_t out_width = w;
    uint32_t out_height = h;
    uint32_t num_src_ptrs = getPixFmtPlaneNum(src_fmt) * batch_size;
    uint32_t num_dst_ptrs = getPixFmtPlaneNum(dst_fmt) * batch_size;
    uint32_t num_dst2_ptrs = getPixFmtPlaneNum(dst2_fmt) * batch_size;

    cncvImageDescriptor src_desc;
    src_desc.width = in_width;
    src_desc.height = in_height;
    src_desc.depth = src_dtype;
    src_desc.pixel_fmt = src_fmt;
    src_desc.color_space = color_space;
    src_desc.stride[0] = 3 * in_width * getSizeOfDepth(src_desc.depth);

    cncvImageDescriptor dst_desc;
    dst_desc.width = out_width;
    dst_desc.height = out_height;
    dst_desc.depth = dst_dtype;
    dst_desc.pixel_fmt = dst_fmt;
    dst_desc.color_space = color_space;
    dst_desc.stride[0] = 3 * out_width * getSizeOfDepth(dst_desc.depth);

    cncvImageDescriptor dst2_desc;
    dst2_desc.width = out_width;
    dst2_desc.height = out_height;
    dst2_desc.depth = dst2_dtype;
    dst2_desc.pixel_fmt = dst2_fmt;
    dst2_desc.color_space = color_space;
    dst2_desc.stride[0] = 3 * out_width * getSizeOfDepth(dst2_desc.depth);

    cncvImageDescriptor *src_descs = nullptr, *dst_descs = nullptr;
    src_descs = (cncvImageDescriptor *)malloc(batch_size * sizeof(cncvImageDescriptor));
    dst_descs = (cncvImageDescriptor *)malloc(batch_size * sizeof(cncvImageDescriptor));
    cncvRect *src_rois = nullptr, *dst_rois = nullptr;
    src_rois = (cncvRect *)malloc(batch_size * sizeof(cncvRect));
    dst_rois = (cncvRect *)malloc(batch_size * sizeof(cncvRect));
    // img descriptor & rectRois
    for (int i = 0; i < batch_size; i++)
    {
        src_descs[i] = src_desc;
        dst_descs[i] = dst_desc;
        src_rois[i].x = 0;
        src_rois[i].y = 0;
        src_rois[i].w = src_descs[i].width;
        src_rois[i].h = src_descs[i].height;
        dst_rois[i].x = 0;
        dst_rois[i].y = 0;
        dst_rois[i].w = dst_descs[i].width;
        dst_rois[i].h = dst_descs[i].height;
    }

    cncvRect dst2_roi;
    dst2_roi.x = 0;
    dst2_roi.y = 0;
    dst2_roi.h = out_height;
    dst2_roi.w = out_width;

    uint64_t total_src_data_size = getVariableBatchDataSize(batch_size, src_descs);
    uint64_t total_dst_data_size = getVariableBatchDataSize(batch_size, dst_descs);
    uint64_t total_dst2_data_size = getFixedBatchDataSize(batch_size, dst2_desc);

    uint64_t max_image_size = MAX(getMaxImageDataSize(batch_size, src_descs),
                                  getImageDataSize(dst2_desc));
    img_cpu_buffer = malloc(max_image_size);
    // worksize
    size_t workspace_resize, min_workspace_size, total_workspace_size;
    callCNCVFunc(cncvGetResizeWorkspaceSize(batch_size,
                                            src_descs,
                                            src_rois,
                                            dst_descs,
                                            dst_rois,
                                            CNCV_INTER_BILINEAR,
                                            &workspace_resize));

    size_t channel_num = 3;
    callCNCVFunc(cncvGetMeanStdWorkspaceSize(channel_num, &min_workspace_size));
    total_workspace_size = workspace_resize + min_workspace_size;
    workspace = mallocDevice(total_workspace_size);

    cpu_src_ptrs = (void **)malloc(num_src_ptrs * sizeof(void *));
    cpu_dst_ptrs = (void **)malloc(num_dst_ptrs * sizeof(void *));
    cpu_dst2_ptrs = (void **)malloc(num_dst2_ptrs * sizeof(void *));

    mlu_src_ptrs = (void **)mallocDevice(num_src_ptrs * sizeof(void *));
    mlu_dst_ptrs = (void **)mallocDevice(num_dst_ptrs * sizeof(void *));
    mlu_dst2_ptrs = (void **)mallocDevice(num_dst2_ptrs * sizeof(void *));

    mlu_src_datas = mallocDevice(total_src_data_size);
    mlu_dst_datas = mallocDevice(total_dst_data_size + total_dst2_data_size);
    callCNRTFunc(cnrtMemset((uint8_t *)mlu_dst_datas + total_dst_data_size, 0, total_dst2_data_size));

    uint64_t src_data_offset = 0;
    uint64_t dst_data_offset = 0;
    uint64_t dst2_data_offset = 0;

    for (uint32_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy((uint8_t *)mlu_src_datas + src_data_offset,
                                src_mat.data,
                                src_desc.stride[0] * src_desc.height,
                                CNRT_MEM_TRANS_DIR_HOST2DEV));

        cpu_src_ptrs[i] = (uint8_t *)mlu_src_datas + src_data_offset;
        cpu_dst_ptrs[i] = (uint8_t *)mlu_dst_datas + dst_data_offset;
        cpu_dst2_ptrs[i] = (uint8_t *)mlu_dst_datas + dst2_data_offset + total_dst_data_size;

        src_data_offset += src_desc.stride[0] * src_desc.height;
        dst_data_offset += dst_desc.stride[0] * dst_desc.height;
        dst2_data_offset += dst2_desc.stride[0] * dst2_desc.height;
    }

    callCNRTFunc(cnrtMemcpy(mlu_src_ptrs,
                            cpu_src_ptrs,
                            num_src_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    callCNRTFunc(cnrtMemcpy(mlu_dst_ptrs,
                            cpu_dst_ptrs,
                            num_dst_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    callCNRTFunc(cnrtMemcpy(mlu_dst2_ptrs,
                            cpu_dst2_ptrs,
                            num_dst2_ptrs * sizeof(void *),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));

    callCNCVFunc(
        cncvResize_AdvancedROI(handle,
                               batch_size,
                               src_descs,
                               src_rois,
                               mlu_src_ptrs,
                               dst_descs,
                               dst_rois,
                               mlu_dst_ptrs,
                               workspace_resize,
                               (uint8_t *)workspace,
                               CNCV_INTER_BILINEAR));
    float mean[3] = {0, 0, 0};
    float std[3] = {255.0, 255.0, 255.0};
    callCNCVFunc(cncvMeanStd(handle,
                             batch_size,
                             dst_desc,
                             mlu_dst_ptrs,
                             mean,
                             std,
                             dst2_desc,
                             mlu_dst2_ptrs,
                             min_workspace_size,
                             (uint8_t *)workspace + workspace_resize));
    callCNRTFunc(cnrtQueueSync(queue));

    // for (size_t i = 0; i < batch_size; i++)
    // {
    //     callCNRTFunc(cnrtMemcpy(img_cpu_buffer,
    //                             cpu_dst_ptrs[i],
    //                             dst_desc.stride[0] * dst_desc.height,
    //                             CNRT_MEM_TRANS_DIR_DEV2HOST));
    //     std::string target_name = "./test/resize_0" + std::to_string(i) + ".png";
    //     saveDstImage(img_cpu_buffer, target_name, dst_desc, CV_8UC3);
    //     // printMat((uchar *)img_cpu_buffer, 490);
    // }
    for (size_t i = 0; i < batch_size; i++)
    {
        callCNRTFunc(cnrtMemcpy(data,
                                cpu_dst2_ptrs[i],
                                dst2_desc.stride[0] * dst2_desc.height,
                                CNRT_MEM_TRANS_DIR_DEV2HOST));

        // printMat((float *)data, 490);
    }
    std::cout << std::endl;
}

CNCVpreprocess::~CNCVpreprocess()
{
    release();
}
