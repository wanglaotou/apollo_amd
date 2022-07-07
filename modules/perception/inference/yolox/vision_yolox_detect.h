#ifndef SSD_DETECTOR_H_
#define SSD_DETECTOR_H_

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// #include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "rect_class_score.h"

#include "autoware_yolox.h"
#include "tinyxml2.h"

#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <iostream>
#include <sys/stat.h> 
#include <unistd.h>  
#include <dirent.h>  
#include <stdlib.h>  
#include <cstdlib>
#include <math.h>
#include <numeric>

class Monitor_YOLOX
{
public:
    // Monitor_YOLOX();
    // ~Monitor_YOLOX();
    // int init(const char* config_file_path,int whichGPU);
    Monitor_YOLOX(const char* config_file_path,int whichGPU);
    // int work(const char* video_path, const char* choose_mode);
    // step1：det and seg part
    // std::vector<rmRECT>  work(const cv::Mat rawimg);
    // cv::Mat  work_seg(const cv::Mat rawimg);
    // step2：det and seg both
    int work_all(const cv::Mat rawimg,  std::vector<rmRECT> &DetectiontRects, cv::Mat &resultSeg, cv::Mat &inputSeg);

private:
    char m_cfgpath[256];
    int m_whichGPU;

    std::unique_ptr<TrtAutoYolox> m_pAutoYolox;

    int autoYoloxInit(tinyxml2::XMLElement* modelNode);           // 模型初始化
};

#endif //YOLOX_DETECTOR_H
