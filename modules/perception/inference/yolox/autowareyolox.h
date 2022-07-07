/*
 * @Author: your name
 * @Date: 2021-09-16 10:02:17
 * @LastEditTime: 2021-09-27 10:29:06
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WEBAI_SERVER/include/monitors/gdcyMonitor.h
 */
#ifndef _MONITOR_GDCY_H
#define _MONITOR_GDCY_H
#include "autoware_yolox.h"
#include "tinyxml2.h"

#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <sys/stat.h> 
#include <unistd.h>  
#include <dirent.h>  
#include <stdlib.h>  
#include <cstdlib>
#include <string>
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
    int work(cv::Mat rawimg,  std::vector<rmRECT> DetectiontRects);

private:
    char m_cfgpath[256];
    int m_whichGPU;

    std::vector<rmRECT> DetectiontRects;

    std::unique_ptr<TrtAutoYolox> m_pAutoYolox;

    int autoYoloxInit(tinyxml2::XMLElement* modelNode);           // 模型初始化
    // float judgeBySegresult(cv::Mat segResult);                  // 通过分割结果判断是否报警
};


#endif //_MONITOR_GDCY_H