#ifndef TENSORRT_LOGGER_H
#define TENSORRT_LOGGER_H

#include "NvInfer.h"
#include <iostream>

class Logger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) override{
        if(severity == Severity::kERROR)
            std::cout<<msg<<std::endl;
    }
};

#endif //TENSORRT_LOGGER_H