#ifndef PLUGIN_FACTORY_HPP
#define PLUGIN_FACTORY_HPP

// #include "Trt.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"

#include <map>


using namespace nvinfer1;


// integration for serialization
class PluginFactory : public nvcaffeparser1::IPluginFactoryV2 {
public:
    PluginFactory();

    virtual ~PluginFactory() {}
    // ------------------inherit from IPluginFactoryV2--------------------
    // determines if a layer configuration is provided by an IPluginV2
    virtual bool isPluginV2(const char* layerName) override;

    // create a plugin
    // virtual IPluginV2* createPlugin(const char* layerName, const Weights* weights, int nbWeights, const char* libNamespace="") override;

};


#endif