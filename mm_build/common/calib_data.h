/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: A derived implement for CalibDataInterface.
 *************************************************************************/
#pragma once
#include <string>
#include <vector>
#include "mm_calibrator.h"
/*
 * Class SampleCalibData supports three types of initialization:
 * 0. Read binary files from data_path with one static shape. One path represents one batch of input.
 * 1. Read binary files from data_path with dynamic shapes. One path represents one batch of input.
 * 2. Use random data from min-max range to fill calibration input.
 * 3. Use zeros for only one iteration of calibration.
 */
class SampleCalibData : public magicmind::CalibDataInterface
{
public:
    SampleCalibData(const magicmind::Dims &shape,
                    const magicmind::DataType &data_type,
                    int max_samples,
                    const std::vector<std::string> &data_paths);

    SampleCalibData(const std::vector<magicmind::Dims> &shapes,
                    const magicmind::DataType &data_type,
                    int max_samples,
                    const std::vector<std::string> &data_paths);

    SampleCalibData(const magicmind::Dims &shape,
                    const magicmind::DataType &data_type,
                    int max_samples,
                    float min,
                    float max);

    SampleCalibData(const magicmind::Dims &shape, const magicmind::DataType &data_type);

    ~SampleCalibData() { free(buffer_); }

    magicmind::Dims GetShape() const override final;
    magicmind::DataType GetDataType() const override final;
    magicmind::Status Next() override final;
    magicmind::Status Reset() override final;
    void *GetSample() override final;

private:
    bool FillFromFile();
    bool FillFromRand();
    void Init();
    void MoveShapeNext();

private:
    std::vector<magicmind::Dims> shapes_ = {};
    std::vector<int> batch_sizes_ = {};
    int current_shape_idx_ = -1;
    magicmind::DataType data_type_;
    int max_samples_ = -1;
    std::vector<std::string> data_paths_ = {};
    float min_ = 0;
    float max_ = 0;
    bool use_rand_ = false;
    void *buffer_ = nullptr;
    int current_sample_ = 0;
};
