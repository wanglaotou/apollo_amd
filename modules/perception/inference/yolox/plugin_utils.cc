/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2019-12-09 11:09:34
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-09-14 14:52:48
 */
#include "plugin_utils.h"

size_t type2size(nvinfer1::trtDataType type) {
    if(type == nvinfer1::trtDataType::kFLOAT) {
        return 4;
    } else if (type == nvinfer1::trtDataType::kHALF) {
        return 2;
    } else if (type == nvinfer1::trtDataType::kINT8) {
        return 1;
    } else {
        ASSERT(false);
    }
}

void* copyToDevice(const void* data, size_t count) {
    void *deviceData;
    CUDA_CHECK(cudaMalloc(&deviceData, count));
    CUDA_CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
    return deviceData;
}

void copyToBuffer(char*& buffer, const void* data, size_t count) {
    memcpy(buffer, data, count);
}

void convertAndCopyToDeivce(void*& deviceWeights, const nvinfer1::Weights &weights,
                            nvinfer1::trtDataType trtdatatype) {
    size_t size = weights.count * type2size(trtdatatype);
    if (weights.type != trtdatatype) // Weights are converted in host memory first, if the type does not match
    {
        void *buffer = malloc(size);
        for (int64_t v = 0; v < weights.count; ++v)
            if (trtdatatype == nvinfer1::trtDataType::kFLOAT)
                static_cast<float *>(buffer)[v] = __half2float(static_cast<const __half *>(weights.values)[v]);
            else
                static_cast<__half *>(buffer)[v] = __float2half(static_cast<const float *>(weights.values)[v]);

        deviceWeights = copyToDevice(buffer, size);
        free(buffer);
    }
    else
        deviceWeights = copyToDevice(weights.values, size);
}

void convertAndCopyToBuffer(char*& buffer, const nvinfer1::Weights weights,
                            nvinfer1::trtDataType trtdatatype) {
    size_t size = weights.count * type2size(trtdatatype);
    if(weights.type != trtdatatype) {
        for (int64_t v = 0; v < weights.count; ++v) { 
        if (trtdatatype == nvinfer1::trtDataType::kFLOAT)
            reinterpret_cast<float *>(buffer)[v] = __half2float(static_cast<const __half *>(weights.values)[v]);
        else
            reinterpret_cast<__half *>(buffer)[v] = __float2half(static_cast<const float *>(weights.values)[v]);
        }
    } else {
        copyToBuffer(buffer, weights.values, size);
    }
    buffer += size;
}