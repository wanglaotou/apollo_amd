/*
 * @Author: your name
 * @Date: 2021-09-14 15:23:10
 * @LastEditTime: 2021-09-29 11:37:13
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WEBAI_SERVER/src/infers/trt_classify.cpp
 */
#include "trt_baseinfer.h"
#include "log.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "omp.h"

BasicTrtInfer::BasicTrtInfer()
{

}
// TODO
BasicTrtInfer::~BasicTrtInfer()
{
    m_pbuffer.release();
}

bool BasicTrtInfer::build()
{
    cudaSetDevice(mParams.gpuIdx);
    initLibNvInferPlugins(&gLogger, "");
    
    TRACE(LOG_INFO, "Try to load trt file\n");
    std::fstream existEngine;
    existEngine.open(mParams.engileFileName, std::ios::in);
    if(existEngine)  //加载序列化模型
    {
        existEngine.close();
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(mParams.engileFileName, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());

        runtime->destroy();
        TRACE(LOG_INFO,"Engine load from: %s\n",mParams.engileFileName.c_str());
        if (!mEngine){
            return false;
        } 
        mcontext = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!mcontext){
            return false;
        }
        m_pbuffer.reset(new samplesCommon::BufferManager(mEngine, mParams.batchSize));
        return true;
    }
    existEngine.close();
    TRACE(LOG_INFO,"No trt file: %s\n",mParams.engileFileName.c_str());

    TRACE(LOG_INFO,"Try to load onnx file\n");
    std::fstream existOnnx;
    existOnnx.open(mParams.onnxFileName, std::ios::in);
    if(existOnnx)  //加载onnx模型
    {
        existOnnx.close();
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        if (!builder){
            return false;
        }
        // bool hasTf32 = builder->platformHasFastFp16

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network){
            return false;
        }
        
        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config){
            return false;
        }

        auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser){
            return false;
        }

        auto parsed = parser->parseFromFile(
            mParams.onnxFileName.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
        if (!parsed){
            return false;
        }

        builder->setMaxBatchSize(mParams.batchSize);
        //!< 设置engine可用最大GPU内存
        config->setMaxWorkspaceSize(1000_MiB);
        if (mParams.fp16){
            config->setFlag(BuilderFlag::kFP16);
        }
        if (mParams.int8){
            config->setFlag(BuilderFlag::kINT8);
            samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
        }

        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        if (!mEngine){
            return false;
        }
        if (mParams.engileFileName.size() > 0){
            std::ofstream p(mParams.engileFileName, std::ios::binary);
            if (!p){
                return false;
            }
            nvinfer1::IHostMemory* ptr = mEngine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
            p.close();
            TRACE(LOG_INFO,"Engine file saved to: %s\n",mParams.engileFileName.c_str());
        }
        //TODO 完成模型序列化之后，删除模型

        mcontext = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!mcontext){
            return false;
        }

        // assert(network->getNbInputs() == 1);
        // nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
        // assert(inputDims.nbDims == 4 || inputDims.nbDims == 3);
        // assert(network->getNbOutputs() == 1);
        m_pbuffer.reset(new samplesCommon::BufferManager(mEngine, mParams.batchSize));
        return true;
    }
    existOnnx.close();
    TRACE(LOG_INFO,"No onnx file: %s\n",mParams.onnxFileName.c_str());

    TRACE(LOG_INFO,"Try load caffe file\n");
    std::fstream existCaffe;
    existCaffe.open(mParams.prototxtFileName, std::ios::in);

    if(existCaffe)  //加载caffe模型
    {
        existCaffe.close();
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        if (!builder){
            return false;
        }


        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
        if (!network){
            return false;
        }


        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config){
            return false;
        }


        auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
        if (!parser){
            return false;
        }

        
        const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(mParams.prototxtFileName.c_str(),
                                                                                    mParams.weightsFileName.c_str(), 
                                                                                    *network, 
                                                                                    trtDataType::kFLOAT);
        //!< 标记输出层
        for (auto& s : mParams.outputTensorNames){
            network->markOutput(*blobNameToTensor->find(s.c_str()));
        }



        builder->setMaxBatchSize(mParams.batchSize);
        //!< 设置engine可用最大GPU内存
        config->setMaxWorkspaceSize(5000_MiB);
        if (mParams.fp16){
            config->setFlag(BuilderFlag::kFP16);
        }


        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        if (!mEngine){
            return false;
        }

        if (mParams.engileFileName.size() > 0){
            std::ofstream p(mParams.engileFileName, std::ios::binary);
            if (!p){
                return false;
            }
            nvinfer1::IHostMemory* ptr = mEngine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
            p.close();
            TRACE(LOG_INFO,"Engine file saved to: %s\n",mParams.engileFileName.c_str());
        }

        
        mcontext = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!mcontext){
            return false;
        }


        // assert(network->getNbInputs() == 1);
        // nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
        // assert(inputDims.nbDims == 3);
        m_pbuffer.reset(new samplesCommon::BufferManager(mEngine, mParams.batchSize));
        return true;
    }
    existCaffe.close();
    TRACE(LOG_INFO,"No caffe file: %s\n",mParams.prototxtFileName.c_str());

    return false;
    
}

bool BasicTrtInfer::infer(const float* iptdata, std::vector<TRT_TENSOR_ATTR_S>& blobtensors)
{
    // samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    if (!mcontext)
    {
        TRACE(LOG_ERROR,"%d: failed context\n", __LINE__);
        return false;
    }

    
    assert(mParams.inputTensorNames.size() == 1);

    const int iptN = mParams.batchSize;
    const int iptC = mParams.inputDim[0];
    const int iptH = mParams.inputDim[1];
    const int iptW = mParams.inputDim[2];
   
    
    // step1：输入预处理，对输入进行减均值除方差操作

    // float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    float* hostDataBuffer = static_cast<float*>(m_pbuffer->getHostBuffer(mParams.inputTensorNames[0]));

    int volImg = iptC * iptH * iptW;
    int volChl = iptH * iptW;
//     omp_set_num_threads(4);
// #pragma omp parallel for
    for(int i = 0; i < iptN; ++i){
        for(int c = 0; c < iptC; ++c){
            for (int j = 0; j < volChl; ++j){
                hostDataBuffer[i * volImg + c * volChl + j] = (iptdata[i*volImg+j*iptC+c] - mParams.pixelMean[c]) * mParams.pixelScale;
            }
        }
    }


    // buffers.copyInputToDevice();
    m_pbuffer->copyInputToDevice();
    // 对于implicit-batch-network采用enqueue/execute，对于explicit-batch-network采用enqueueV2/executeV2
    bool status = false;
    if(true == mEngine->hasImplicitBatchDimension()){
        // status = mcontext->execute(mParams.batchSize, buffers.getDeviceBindings().data());
        status = mcontext->execute(mParams.batchSize, m_pbuffer->getDeviceBindings().data());
    }else{
        // status = mcontext->executeV2(buffers.getDeviceBindings().data());
        status = mcontext->executeV2(m_pbuffer->getDeviceBindings().data());
    }
    if (!status){
        TRACE(LOG_ERROR,"%d: ailed executeV2\n", __LINE__);
        return false;
    }


    // buffers.copyOutputToHost();
    m_pbuffer->copyOutputToHost();

// #pragma omp parallel for
    for(int i = 0; i < iptN; i++)
    {
        for(size_t j = 0; j < mParams.outputTensorNames.size(); j++)
        {
            TRT_TENSOR_ATTR_S tmpOut;
            tmpOut.blobname = mParams.outputTensorNames[j].c_str();
            // tmpOut.num_elems = buffers.size(mParams.outputTensorNames[j]);
            // tmpOut.data = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[j]));
            tmpOut.num_elems = m_pbuffer->size(mParams.outputTensorNames[j])/sizeof(float);
            tmpOut.data = static_cast<const float*>(m_pbuffer->getHostBuffer(mParams.outputTensorNames[j]));
            blobtensors[i*mParams.outputTensorNames.size()+j] = tmpOut;

            // printf("j, blobtensors=%d,%d\n", j, tmpOut.num_elems);      // num_elems=NxCxHxW
            // j, blobtensors=0,5760    --> 865: 1*3*32*60  -> cls
            // j, blobtensors=1,7680    --> 866: 1*4*32*60  -> box
            // j, blobtensors=2,1920    --> 867: 1*1*32*60  -> objectness
            // j, blobtensors=3,1440    --> 880: 1*3*16*30
            // j, blobtensors=4,1920
            // j, blobtensors=5,480
            // j, blobtensors=6,360
            // j, blobtensors=7,480
            // j, blobtensors=8,120
            // j, blobtensors=9,96
            // j, blobtensors=10,128
            // j, blobtensors=11,32
            // j, blobtensors=12,24
            // j, blobtensors=13,32
            // j, blobtensors=14,8
            // j, blobtensors=15,6
            // j, blobtensors=16,8
            // j, blobtensors=17,2
        }
    }
    return true;

}
