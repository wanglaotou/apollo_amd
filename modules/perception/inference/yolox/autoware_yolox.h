/*
 * @Author: your name
 * @Date: 2021-09-15 16:35:04
 * @LastEditTime: 2021-09-22 15:43:38
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WEBAI_SERVER/include/infers/gdcySeg.h
 */
#ifndef _TRT_GDCYSEG_H
#define _TRT_GDCYSEG_H

#include <math.h>
#include <algorithm>
#include <vector>
#include "trt_baseinfer.h"

struct autoYoloxParam : public BASIC_TRT_PARAMS{
    std::string colorType{"RGB"};                       //!< 色彩空间类型
    int outputClsSize{21};                               //!< The number of output classes
    int outputdetClsSize{21};                               //!< The number of output det classes
    int outputsegClsSize{21};                               //!< The number of output seg classes
    int keepTopK{100};                                  //!< The maximum number of detection post-NMS
    float visualThreshold{0.1};                         //!< The minimum score threshold to consider a detection
};

struct rmRECT{
    int top;
    int left;
    int right;
    int bottom;
    float score;
    int classnum;
};

struct _NNIE_STACK_S{
    int s32Min;
    int s32Max;
};

class TrtAutoYolox : public BasicTrtInfer
{
public:
    TrtAutoYolox(const autoYoloxParam& params);
    ~TrtAutoYolox();
    bool run(const cv::Mat ipt, cv::Mat& resultSeg);
    bool run(const cv::Mat ipt, std::vector<std::vector<float> > &ssdrects);
    bool run(const cv::Mat ipt, std::vector<rmRECT> &DetectiontRects);
    bool run(const cv::Mat ipt, std::vector<rmRECT> &DetectiontRects, cv::Mat& resultSeg, cv::Mat& inputSeg);

    float sigmoid(float x);
    int GenerateMeshgrid();

private:
    autoYoloxParam mTaskParams;
    
    // 检测分支
    std::vector<float>meshgrid;
    const int class_num = 8;
    // const int class_num = 6;
    int headNum = 6;

    int input_w = 832;
    int input_h = 480;		

    int strides[6] = {8, 16, 32, 64, 128, 256};//{256, 128, 64, 32, 16,8 }; //
    int mapSize[6][2] = { {60, 104}, {30, 52}, {15, 26}, {8, 13},{4, 7}, {2, 4}};//{{1, 2}, {2, 4}, {4, 8}, {8, 15}, {16, 30}, {32, 60}};

    float nmsThresh = 0.51;
    float objectThresh[8] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5};     //CLASSES = ('car', 'car_reg', 'car_big_reg', 'car_front', 'person')
    // float objectThresh[6] = {0.35, 0.35, 0.35, 0.35, 0.35, 0.35};     //CLASSES = ('person', 'bicyclist', 'motorcyclist', 'car', 'bus', 'truck')

    _NNIE_STACK_S pstStack[2000];

};

#endif //_TRT_GDCYSEG_H