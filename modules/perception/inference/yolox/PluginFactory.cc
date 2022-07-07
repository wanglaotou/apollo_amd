/*
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 11:55:03
 * @LastEditTime: 2021-09-14 14:52:56
 * @LastEditors: Please set LastEditors
 */

#include "PluginFactory.h"
#include "PReLUPlugin.h"
#include <algorithm>
#include <cassert>
#include <iostream>

PluginFactory::PluginFactory() {
}

bool PluginFactory::isPluginV2(const char* layerName) 
{
    std::string strName{layerName};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);
    return (strName.find("prelu") != std::string::npos);
}

// IPluginV2* PluginFactory::createPlugin(const char *layerName, const Weights* weights, int nbWeights, const char* libNamespace) 
// {
//     assert(isPluginV2(layerName));

//     std::string strName{layerName};
//     std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);

//     if (strName.find("prelu") != std::string::npos) {
//         // std::cout << "nbWeight: " << nbWeights << std::endl;
//         // std::cout << "weights.count: " << weights->count << std::endl;
//         return (IPluginV2*)(new PReLUPlugin(weights, nbWeights));
//     }
//     else
//     {
//         std::cout << "warning : " << layerName << std::endl;
//         assert(0);
//         return nullptr;
//     }
// }


