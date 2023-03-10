#include "cncv.h"
#include "cncvutil.h"

class CNCVpreprocess
{
    void **cpu_src_ptrs = nullptr;
    void **cpu_dst_ptrs = nullptr;
    void **cpu_dst2_ptrs = nullptr;
    void **cpu_dst3_ptrs = nullptr;

    void **mlu_src_ptrs = nullptr;
    void **mlu_dst_ptrs = nullptr;
    void **mlu_dst2_ptrs = nullptr;
    void **mlu_dst3_ptrs = nullptr;

    void *mlu_src_datas = nullptr;
    void *mlu_dst_datas = nullptr;
    // void *mlu_dst2_datas = nullptr;
    void *img_cpu_buffer = nullptr;
    void *workspace = nullptr;
    cnrtQueue_t queue = nullptr;
    cncvHandle_t handle = nullptr;

    int device_id = 0;

public:
    CNCVpreprocess(int device);
    void initial();
    void printMat(uint8_t *mat, size_t length);
    void printMat(float *mat, size_t length);
    void release();
    void classPreprocess(cv::Mat &img, float *data, int w, int h);
    void cncvsplit();
    void cncvresize();
    void cncvresizestdsplit(cv::Mat &img, float *data, int w, int h);
    void cncvresizestd(cv::Mat &img, void *data, int w, int h);
    void cncvbgr2rgb();
    ~CNCVpreprocess();
};