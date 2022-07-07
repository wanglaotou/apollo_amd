/*
 * @Author: your name
 * @Date: 2021-09-14 15:10:23
 * @LastEditTime: 2021-09-29 13:57:26
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WEBAI_SERVER/include/infers/trt_classify.h
 */
#ifndef TRT_BASEINFER_H
#define TRT_BASEINFER_H

#include "common.h"
#include "logger.h"
#include "params.h"
#include "buffers.h"

#include "NvOnnxParser.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "PluginFactory.h"


typedef struct TRT_TENSOR_ATTR_S
{
    const float*                   data;  // tensor保存数据buffer，数据按照NCHW格式存放
    unsigned int        num_elems;  // tensor数据元素个数，num_elems=NxCxHxW
    const char*                blobname;  // tensor对应网络中blobname
} TRT_TENSOR_ATTR_S;



class BasicTrtInfer{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
public:
    BasicTrtInfer();
    BasicTrtInfer(const BASIC_TRT_PARAMS& params)
        : mParams(params)
        , mEngine(nullptr)
    {}
    ~BasicTrtInfer();

    bool build();
    //!< 模型推理：均值方差预处理->获取网络输出结果
    bool infer(const float* iptdata, std::vector<TRT_TENSOR_ATTR_S>& blobtensors); //!<blobtensorrs大小需预先设定

protected:
    BASIC_TRT_PARAMS mParams;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network
    SampleUniquePtr<nvinfer1::IExecutionContext> mcontext{nullptr}; //!< 用于推理的上下文
private:
    Logger gLogger;//!< 调用TRTAPI所必须参数
    std::unique_ptr<samplesCommon::BufferManager> m_pbuffer{nullptr};

};

#endif //TRT_BASEINFER_H