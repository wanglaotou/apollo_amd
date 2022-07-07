/*
 * @Author: your name
 * @Date: 2021-09-14 14:51:07
 * @LastEditTime: 2021-09-29 13:46:31
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WEBAI_SERVER/include/commons/params.h
 */
#ifndef _PARAMS_H
#define _PARAMS_H

#include <string>
#include <vector>
#include <iostream>

typedef struct BASIC_TRT_PARAMS
{
    int batchSize{1};                                   //!< Number of inputs in a batch
    int dlaCore{-1};                                    //!< Specify the DLA core to run network on.
    bool int8{false};                                   //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                                   //!< Allow running the network in FP16 mode.
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    int inputDim[3]{3,112,112};                         //!< 输入维度，[C,H,W]
    int gpuIdx{0};                                      //!< 使用gpu序号，从0开始
    float pixelMean[3]{127.5f, 127.5f, 127.5f};         //!< 输入均值，BGR顺序
    float pixelScale{0.007843};                         //!< 输入方差
    std::string engileFileName;                         //!< 序列化模型保存路径，若为空则不保存序列化
    std::string onnxFileName;                           //!< Filename of ONNX file of a network
    std::string prototxtFileName;                       //!< Filename of prototxt design file of a network
    std::string weightsFileName;                        //!< Filename of prototxt design file of a network
} BASIC_TRT_PARAMS;


typedef enum //云端初始化报警
{
    RM_RES_AUTOWARE_BOX_NORMAL             = 17,       ///< Autoware boxdet判断检测框正常
    RM_RES_AUTOWARE_BOX_NOTDETECT          = 16,       ///<Autoware boxdet判断无检测框，nodet
    RM_RES_AUTOWARE_BOX_MOVE               = 15,       ///< Autoware boxdet判断检测框过低（镜头异常），move
    RM_RES_MUCKTRUCK_BOX_NORMAL             = 14,       ///< 渣土boxdet判断检测框正常
    RM_RES_MUCKTRUCK_BOX_NOTDETECT          = 13,       ///< 渣土boxdet判断无检测框，nodet
    RM_RES_MUCKTRUCK_BOX_MOVE               = 12,       ///< 渣土boxdet判断检测框过低（镜头异常），move
    RM_RES_MUCKTRUCK_CAMERA_MOVE            = 11,       ///< 渣土camera判断为镜头异常，move
    RM_RES_MUCKTRUCK_CAMERA_NORMAL          = 10,       ///< 渣土camera判断为正常，normal
    RM_RES_MUCKTRUCK_CAMERA_MUSK            = 9,        ///< 渣土camera判断为污染，musk
    RM_RES_MUCKTRUCK_ZHATU                  = 8,        ///< 渣土earthtype判断为渣土，zhatu
    RM_RES_MUCKTRUCK_BUILD                  = 7,        ///< 渣土earthtype判断为建筑垃圾，building
    RM_RES_MUCKTRUCK_CLOSE                  = 6,        ///< 渣土close判断为密闭，close
    RM_RES_MUCKTRUCK_OPEN                   = 5,        ///< 渣土close判断为开厢，open
    RM_RES_MUCKTRUCK_LOAD                   = 4,        ///< 渣土load判断为装载，load
    RM_RES_MUCKTRUCK_EMPTY                  = 3,        ///< 渣土load判断为空厢，empty

    RM_RES_ALLERT_FALSE                     = 2,        ///< 云端算法认为误报，过滤
    RM_RES_ALLERT_TRUE                      = 1,        ///< 云端算法认为正报，不过滤

    RM_RES_SUCCESS                          = 0,        ///< 成功
    RM_RES_ERROR_MODEL_PATH                 = -1,       ///< 模型路径加载错误
    RM_RES_ERROR_PARSE_CONFIG               = -2,       ///< 配置文件解析错误
    RM_RES_ERROR_VERSION_MISMATCH           = -3,       ///< 版本号不匹配
    RM_RES_ERROR_MODEL_INIT_FAILED          = -4,       ///< 模型初始化失败
    RM_RES_ERROR_VIDEO_POOR                 = -5,       ///< 视频分辨率低于标准
    RM_RES_ERROR_VIDEO_TYPE                 = -6,       ///< 视频格式存在问题
    RM_RES_ERROR_VIDEO_PATH                 = -7,       ///< 视频路径存在问题

    RM_RES_ERROR_PICTURE_OPEN               = -8,       ///< 图片无法打开
    RM_RES_ERROR_PICTURE_SIZE               = -9,       ///< 图片尺寸不达标
    
    RM_RES_ERROR_MKDIR_FAILED               = -10,      ///< 路径创建失败

    RM_RES_ERROR_FUNCTION_NOT_SUPPORT       = -15,      ///< 算法子功能不支持
    RM_RES_ERROR_INFERENCE_FAILED           = -16,      ///< 模型推理失败

}RM_AISERVER_TYPE_E;


#endif // _PARAMS_H