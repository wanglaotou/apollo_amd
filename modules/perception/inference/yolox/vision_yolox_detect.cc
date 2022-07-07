/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vision_yolox_detect.h"
#include "log.h"
#include "params.h"
#include <iostream>
#include <math.h>


Monitor_YOLOX::Monitor_YOLOX(const char* config_file_path,int whichGPU)
{
    // if(NULL == config_file_path)
    // {
    //     TRACE(LOG_ERROR, "%d: model path is %s\n", __LINE__, config_file_path);
    //     return -1;
    // }
    int ret = 0;
    
    strcpy(m_cfgpath, config_file_path);

    m_whichGPU = whichGPU;
    char pXmlpath[256] = {0};
    sprintf(pXmlpath, "%s/config/config.xml", m_cfgpath);
    std::cout<<"pXmlpath:"<<pXmlpath<<std::endl;
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError xmlresult = doc.LoadFile(pXmlpath);
    if(0 != xmlresult)
    {
        TRACE(LOG_ERROR, "%d: failed to load xml file %s\n", __LINE__, pXmlpath);
        // return -2;
    }
    tinyxml2::XMLElement* root = doc.RootElement();
    // tinyxml2::XMLElement* configNode = root->FirstChildElement("config");
    tinyxml2::XMLElement* modelNode = root->FirstChildElement("modelconfig");
    ret = autoYoloxInit(modelNode);
    if(0 != ret)
    {
        TRACE(LOG_ERROR,"%d: autoYoloxInit failed\n", __LINE__);
        // return -1;
    }
    // return RM_RES_SUCCESS;
}


int Monitor_YOLOX::autoYoloxInit(tinyxml2::XMLElement* modelNode)
{
//!< 初始化GDCYSEG网络
    autoYoloxParam gautoyoloxparams;
    tinyxml2::XMLElement* gdcysegNode = modelNode->FirstChildElement("boxdet");
    gautoyoloxparams.gpuIdx = m_whichGPU;
    // gautoyoloxparams.usingModelType = std::stoi(gdcysegNode->FirstChildElement("usingModelType")->GetText());
    gautoyoloxparams.colorType = gdcysegNode->FirstChildElement("colorType")->GetText();
    gautoyoloxparams.inputDim[0] = std::stoi(gdcysegNode->FirstChildElement("inputChannel")->GetText());
    gautoyoloxparams.inputDim[1] = std::stoi(gdcysegNode->FirstChildElement("inputHeight")->GetText());
    gautoyoloxparams.inputDim[2] = std::stoi(gdcysegNode->FirstChildElement("inputWidth")->GetText());
    gautoyoloxparams.pixelMean[0] = std::stof(gdcysegNode->FirstChildElement("inputMeanR")->GetText());
    gautoyoloxparams.pixelMean[1] = std::stof(gdcysegNode->FirstChildElement("inputMeanG")->GetText());
    gautoyoloxparams.pixelMean[2] = std::stof(gdcysegNode->FirstChildElement("inputMeanB")->GetText());
    gautoyoloxparams.pixelScale = std::stof(gdcysegNode->FirstChildElement("inputScale")->GetText());
    gautoyoloxparams.batchSize = std::stoi(gdcysegNode->FirstChildElement("inputBatchsize")->GetText());
    gautoyoloxparams.outputdetClsSize = std::stoi(gdcysegNode->FirstChildElement("outputdetClassSize")->GetText());
    gautoyoloxparams.outputsegClsSize = std::stoi(gdcysegNode->FirstChildElement("outputsegClassSize")->GetText());
    gautoyoloxparams.inputTensorNames.push_back(gdcysegNode->FirstChildElement("inputTensorName")->GetText());
    // TODO
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName1")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName2")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName3")->GetText());

    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName4")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName5")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName6")->GetText());

    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName7")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName8")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName9")->GetText());

    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName10")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName11")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName12")->GetText());

    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName13")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName14")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName15")->GetText());

    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName16")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName17")->GetText());
    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName18")->GetText());

    gautoyoloxparams.outputTensorNames.push_back(gdcysegNode->FirstChildElement("outputTensorName19")->GetText());


    if("BGR" == gautoyoloxparams.colorType){
        float tmp = gautoyoloxparams.pixelMean[0];
        gautoyoloxparams.pixelMean[0] = gautoyoloxparams.pixelMean[2];
        gautoyoloxparams.pixelMean[2] = tmp;
    }

    char pEnginepath[256] = { 0 };
    sprintf(pEnginepath, "%s/%s", m_cfgpath, gdcysegNode->FirstChildElement("serializemodelpath")->GetText());
    gautoyoloxparams.engileFileName = pEnginepath;

    char pRECproto[256] = { 0 };
    char pRECmodel[256] = { 0 };
    sprintf(pRECproto, "%s/%s", m_cfgpath, gdcysegNode->FirstChildElement("caffeprototxt")->GetText());
    sprintf(pRECmodel, "%s/%s", m_cfgpath, gdcysegNode->FirstChildElement("caffemodel")->GetText());
    gautoyoloxparams.prototxtFileName = pRECproto;
    gautoyoloxparams.weightsFileName = pRECmodel;
    printf("pRECproto, pRECmodel=%s,%s\n", pRECproto, pRECmodel);
    
    char pREConnxpath[256] = { 0 };
    sprintf(pREConnxpath, "%s/%s", m_cfgpath, gdcysegNode->FirstChildElement("onnxpath")->GetText());
    gautoyoloxparams.onnxFileName = pREConnxpath;
    
    m_pAutoYolox.reset(new TrtAutoYolox(gautoyoloxparams));

    if(!m_pAutoYolox->build())
    {
        TRACE(LOG_ERROR,"%d: TrtAutoYolox build failed\n",__LINE__);
        return -1;
    }
    TRACE(LOG_INFO,"TrtAutoYolox build success\n");
    return 0;
//!< 初始化GDCY网络
    
}


// det & seg both
int Monitor_YOLOX::work_all(const cv::Mat rawimg,  std::vector<rmRECT> &DetectiontRects, cv::Mat &resultSeg, cv::Mat &inputSeg)
{
    if(!rawimg.data)
    {
        TRACE(LOG_ERROR,"%d: could not open image\n",__LINE__);
    }

    // 获取检测推理结果
    cv::Mat resultDet;
    cv::resize(rawimg, resultDet, cv::Size(832, 480));
    cv::resize(rawimg, resultSeg, cv::Size(208, 120));      // 输入

    if(!m_pAutoYolox->run(rawimg,DetectiontRects,resultSeg, inputSeg))
    {
        TRACE(LOG_ERROR,"%d: infer failed\n",__LINE__);
        // return RM_RES_ERROR_INFERENCE_FAILED;
    }
    TRACE(LOG_INFO,"DetectiontRects info: %d\n",DetectiontRects.size());
    // float scale_width = 1.0; // * 1280 / 832;
    // float scale_height = 1.0; // * 720 / 480;
    for(size_t i=0;i<DetectiontRects.size();i++)
    {
        int left = (DetectiontRects[i].left);
        int top = (DetectiontRects[i].top);
        int right = (DetectiontRects[i].right);
        int bottom = (DetectiontRects[i].bottom);
        // int classnum = DetectiontRects[i].classnum;
        // float score = DetectiontRects[i].score;

        // std::cout<<"DetectiontRects info="<<left<<", "<<top<<", "<<right<<", "<<bottom<<","<<classnum<<std::endl;
        cv::rectangle(resultDet,cv::Rect(left,top,right-left,bottom-top),cv::Scalar(0,0,255),1,1,0);

    }
    cv::imwrite("/home/mario/projects/resultDet2.jpg",  resultDet);

    cv::resize(resultSeg, resultSeg, cv::Size(rawimg.cols, rawimg.rows));
    // std::cout<<"resultSeg shape:"<<resultSeg.size()<<std::endl;
    cv::imwrite("/home/mario/projects/resultSeg2.jpg",  resultSeg);
    // cv::cvtColor(resultSeg, resultSeg, CV_BGR2GRAY);
    // return resultSeg;
    // return DetectiontRects;
    return 1;
}

