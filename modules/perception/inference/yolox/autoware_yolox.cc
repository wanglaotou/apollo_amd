/*
 * @Author: your name
 * @Date: 2021-09-15 16:35:14
 * @LastEditTime: 2021-09-27 10:28:34
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WEBAI_SERVER/src/infers/gdcySeg.cpp
 */
#include "autoware_yolox.h"
#include "log.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "omp.h"

TrtAutoYolox::TrtAutoYolox(const autoYoloxParam& params) : mTaskParams(params)
{
    mParams.batchSize               = params.batchSize;
    mParams.dlaCore                 = params.dlaCore;
    mParams.int8                    = params.int8;
    mParams.fp16                    = params.fp16;
    mParams.inputTensorNames        = params.inputTensorNames;
    mParams.outputTensorNames       = params.outputTensorNames;
    mParams.inputDim[0]             = params.inputDim[0];
    mParams.inputDim[1]             = params.inputDim[1];
    mParams.inputDim[2]             = params.inputDim[2];
    mParams.gpuIdx                  = params.gpuIdx;
    mParams.pixelMean[0]            = params.pixelMean[0];
    mParams.pixelMean[1]            = params.pixelMean[1];
    mParams.pixelMean[2]            = params.pixelMean[2];
    mParams.pixelScale              = params.pixelScale;
    mParams.engileFileName          = params.engileFileName;
    mParams.onnxFileName            = params.onnxFileName;
    mParams.prototxtFileName        = params.prototxtFileName;
    mParams.weightsFileName         = params.weightsFileName;

    // mTaskParams.colorType           = params.colorType;

    mEngine = nullptr;
    mcontext = nullptr;
}

TrtAutoYolox::~TrtAutoYolox()
{
    
}


int TrtAutoYolox::GenerateMeshgrid()
{	
	int ret = 0;
	if(headNum == 0)
	{
		printf("=== YOLOX Meshgrid  Generate failed! \n");
	}
	
    for(int index = 0; index < headNum; index ++)
    {
        for(int i =0; i < mapSize[index][0]; i ++)
        {
            for(int j = 0; j < mapSize[index][1]; j ++)
            {
                meshgrid.push_back(float(j));
                meshgrid.push_back(float(i));
            }
        }
    }

	printf("=== YOLOX Meshgrid  Generate success! \n");

	return ret;
}

static inline float fast_exp(float x)
{
	//return exp(x);
    union {uint32_t i;float f;} v;
  	v.i = (12102203.1616540672*x+1064807160.56887296);
    return v.f;
}

float TrtAutoYolox::sigmoid(float x)
{
	return 1 / (1 + fast_exp(-x));
}

static inline void fcw_Argswap(rmRECT* ps32Src1, rmRECT* ps32Src2)
{
	int i = 0;
	rmRECT u32Tmp = {0};
	for( i = 0; i < 1; i++ )
	{
		u32Tmp = ps32Src1[i];
		ps32Src1[i] = ps32Src2[i];
		ps32Src2[i] = u32Tmp;
	}
}

static inline int fcw_QuickSort(rmRECT* ps32Array,int s32Low, int s32High, _NNIE_STACK_S *pstStack,int u32MaxNum)
{
	int i = s32Low;
	int j = s32High;
	int s32Top = 0;
	float s32KeyConfidence = ps32Array[s32Low].score;
	pstStack[s32Top].s32Min = s32Low;
	pstStack[s32Top].s32Max = s32High;

	while(s32Top > -1)
	{
		s32Low = pstStack[s32Top].s32Min;
		s32High = pstStack[s32Top].s32Max;
		i = s32Low;
		j = s32High;
		s32Top--;

		s32KeyConfidence = ps32Array[s32Low].score;

		while(i < j)
		{
			while((i < j) && (s32KeyConfidence > ps32Array[j].score))
			{
				j--;
			}
			if(i < j)
			{
				fcw_Argswap(&ps32Array[i], &ps32Array[j]);
				i++;
			}

			while((i < j) && (s32KeyConfidence < ps32Array[i].score))
			{
				i++;
			}
			if(i < j)
			{
				fcw_Argswap(&ps32Array[i], &ps32Array[j]);
				j--;
			}
		}

		if(s32Low <= u32MaxNum)
		{
			if(s32Low < i-1)
			{
				s32Top++;
				pstStack[s32Top].s32Min = s32Low;
				pstStack[s32Top].s32Max = i-1;
			}

			if(s32High > i+1)
			{
				s32Top++;
				pstStack[s32Top].s32Min = i+1;
				pstStack[s32Top].s32Max = s32High;
			}
		}
	}
	return 0;
}

static inline float IOU(int s32XMin1, int s32YMin1, int s32XMax1, int s32YMax1, int s32XMin2, int s32YMin2, int s32XMax2, int s32YMax2)
{
	int s32Inter = 0;
	int s32Total = 0;
	int s32XMin = 0;
	int s32YMin = 0;
	int s32XMax = 0;
	int s32YMax = 0;
	int s32Area1 = 0;
	int s32Area2 = 0;
	int s32InterWidth = 0;
	int s32InterHeight = 0;

	s32XMin = MAX(s32XMin1, s32XMin2);
	s32YMin = MAX(s32YMin1, s32YMin2);
	s32XMax = MIN(s32XMax1, s32XMax2);
	s32YMax = MIN(s32YMax1, s32YMax2);

	s32InterWidth = s32XMax - s32XMin + 1;
	s32InterHeight = s32YMax - s32YMin + 1;

	s32InterWidth = ( s32InterWidth >= 0 ) ? s32InterWidth : 0;
	s32InterHeight = ( s32InterHeight >= 0 ) ? s32InterHeight : 0;

	s32Inter = s32InterWidth * s32InterHeight;
	
	s32Area1 = (s32XMax1 - s32XMin1 + 1) * (s32YMax1 - s32YMin1 + 1);
	s32Area2 = (s32XMax2 - s32XMin2 + 1) * (s32YMax2 - s32YMin2 + 1);

	s32Total = s32Area1 + s32Area2 - s32Inter;

	return float(s32Inter)/float(s32Total);
}

bool TrtAutoYolox::run(const cv::Mat ipt, std::vector<rmRECT> &DetectiontRects, cv::Mat& resultSeg, cv::Mat& inputSeg)
{
    // const int inputC = mTaskParams.inputDim[0];
    const int inputH = mTaskParams.inputDim[1];
    const int inputW = mTaskParams.inputDim[2];
    const int batchSize = mTaskParams.batchSize;

    if (1 != batchSize){
        cout <<"only batchsize=1 support"<<endl;
        return false;
    }
    
    cv::Mat resize_ipt;
    // cv::resize(ipt, resize_ipt, cv::Size(inputW, inputH));  //按照网络输入宽高进行resize

    //按照长边等比例缩放
    float scale_h = ipt.rows * 1.0 / inputH;
    float scale_w = ipt.cols * 1.0 / inputW;
    float keep_ratio = std::max(scale_h, scale_w);
    cv::Mat resize_ratio_ipt;
    // cv::Mat resize_ratio_ipt = cv::Mat(cv::Size(inputW, inputH), CV_8UC3, cv::Scalar(114,114,114));
    if(keep_ratio==scale_w)
    {
        int input_w = int(ipt.cols * 1.0 / keep_ratio + 0.5);
        int input_h = int(ipt.rows * 1.0 / keep_ratio + 0.5);
        cv::resize(ipt, resize_ratio_ipt, cv::Size(input_w, input_h));
        int pad_h = inputH - input_h;
        cv::copyMakeBorder(resize_ratio_ipt, resize_ipt, 0, pad_h, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    }
    else
    {
        int input_w = int(ipt.cols * 1.0 / keep_ratio + 0.5);
        int input_h = int(ipt.rows * 1.0 / keep_ratio + 0.5);
        cv::resize(ipt, resize_ratio_ipt, cv::Size(input_w, input_h));
        int pad_w = inputW - input_w;
        cv::copyMakeBorder(resize_ratio_ipt, resize_ipt, 0, 0, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    }

    // cv::imwrite("/home/mario/projects/resize_ipt2.jpg", resize_ipt);

    if("GRAY" == mTaskParams.colorType){
        cv::cvtColor(resize_ipt,resize_ipt,cv::COLOR_BGR2GRAY);
    }else if("RGB" == mTaskParams.colorType){
        cv::cvtColor(resize_ipt,resize_ipt,cv::COLOR_BGR2RGB);
    }

    std::vector<TRT_TENSOR_ATTR_S> blobtensors(mTaskParams.batchSize*mTaskParams.outputTensorNames.size());
    float* iptarray = (float*)malloc(resize_ipt.channels()*sizeof(float)*resize_ipt.cols*resize_ipt.rows);

    if(!resize_ipt.isContinuous())
    {
        TRACE(LOG_ERROR,"%d: img should be continuos, got %s",__LINE__,resize_ipt.isContinuous());
        free(iptarray);
        return false;
    }

    for(int i=0; i<resize_ipt.cols*resize_ipt.rows*resize_ipt.channels(); i++)
    {
        *(iptarray+i) = (float)resize_ipt.data[i];
    }

    bool ret = infer(iptarray, blobtensors);        // tensorrt7/trt_baseinfer.cpp

    free(iptarray);
    if(!ret)
    {
        TRACE(LOG_ERROR,"%d: infer failed",__LINE__);
        return false;
    }

    if(meshgrid.empty())
	{
		ret = GenerateMeshgrid();
	}
    const int maxinum_nms = 300;
	rmRECT res[maxinum_nms] = {0};
	int res_index = 0;

	int gridIndex = -2;

	float cx = 0, cy = 0, xf = 0, yf = 0;
	float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
	float ce_val = 0, cls_val = 0;
    
    for(int p=0; p<batchSize; p++)
    {

        for(int index = 0; index < headNum; index ++)
        {
            const float* cls = blobtensors[index*3+0].data;
            const float* reg = blobtensors[index*3+1].data;
            const float* ce = blobtensors[index*3+2].data;

            // std::cout<<"cls: "<<*cls<<std::endl;     //0x55b14e576f40
            // std::cout<<"objectness: "<<*ce<<std::endl;       //0x55b14e54df30
            // std::cout<<"bbox: "<<*reg<<std::endl;       //0x55b14e38bae0

            
            for(int h = 0; h < mapSize[index][0]; h++)
            {
                for(int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    ce_val = sigmoid(ce[h * mapSize[index][1] + w]);
                    // std::cout<<"ce_val:"<<ce_val<<std::endl;

                    for(int cl = 0; cl < class_num; cl ++)
                    {
                        cls_val = sigmoid(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * ce_val;
                        // std::cout<<"cls_val:"<<cls_val<<std::endl;

                        if(cls_val > objectThresh[cl])
                        {
                            cx = (meshgrid[gridIndex + 0] + reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] ) * strides[index];
                            cy = (meshgrid[gridIndex + 1] + reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index];
                            xf = exp(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] ) * strides[index];
                            yf = exp(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] ) * strides[index];

                            // std::cout<<"cx:"<<cx<<", "<<"cy:"<<cy<<", "<<"xf:"<<xf<<", "<<"yf:"<<yf<<std::endl;

                            xmin = cx - xf / 2;
                            ymin = cy - yf / 2;
                            xmax = cx + xf / 2;
                            ymax = cy + yf / 2;
                            
                            xmin = xmin > 0 ? xmin : 0;
                            ymin = ymin > 0 ? ymin : 0;
                            xmax = xmax < input_w ? xmax : input_w -1;
                            ymax = ymax < input_h ? ymax : input_h -1;
                            // std::cout<<"xmin:"<<xmin<<", "<<"ymin:"<<ymin<<", "<<"xmax:"<<xmax<<", "<<"ymax:"<<ymax<<std::endl;

                            if(xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                            {
                                rmRECT temp;
                                temp.left = int(xmin);
                                temp.top = int(ymin);
                                temp.right = int(xmax);
                                temp.bottom = int(ymax);
                                temp.classnum = cl;
                                temp.score = cls_val;

                                res[res_index++] = temp;
                                if(res_index >= maxinum_nms)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        // return true;
        
        int topK = res_index;
        int topKmin = (res_index < topK ? res_index : topK);
        if(res_index > 0)
        {				
            fcw_QuickSort(res, 0, res_index, pstStack, topKmin);
            
            for(int i = 0; i < topKmin; ++i)
            {
                int xmin1 = res[i].left;
                int ymin1 = res[i].top;
                int xmax1 = res[i].right;
                int ymax1 = res[i].bottom;
                int classId = res[i].classnum;
        
                if(res[i].classnum != -1)
                {
                    rmRECT RectTemp;
                    RectTemp.left = xmin1;
                    RectTemp.top = ymin1;
                    RectTemp.right = xmax1;
                    RectTemp.bottom = ymax1;
                    RectTemp.score = res[i].score;
                    RectTemp.classnum = classId + 1;

                    // std::cout<<"left:"<<xmin1<<", "<<"top:"<<ymin1<<", "<<"right:"<<xmax1<<", "<<"bottom:"<<ymax1<<std::endl;
                    
                    DetectiontRects.push_back(RectTemp);
        
                    for(int j = i + 1; j < topKmin; ++j)
                    {
                        //if(res[j].classnum == classId)
                        //{
                        int xmin2 = res[j].left;
                        int ymin2 = res[j].top;
                        int xmax2 = res[j].right;
                        int ymax2 = res[j].bottom;								
                        float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                        if(iou > nmsThresh)
                        {
                            res[j].classnum = -1;
                        }
                        //}
                    }
                }
            }
        }
    }
    
    int optH = 120;
    int optW = 208; 
    for (int h = 0; h < optH; h++)
    {
        for (int w = 0; w < optW; w++)
        {
            std::vector<float> compv;
            compv.push_back(blobtensors[18].data[0*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[1*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[2*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[3*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[4*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[5*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[6*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[7*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[8*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[9*(optH*optW)+(h*optW+w)]);
            // std::vector<float>::iterator biggest = std::max_element(std::begin(compv),std::end(compv));
            // int max_index = std::distance(std::begin(compv),biggest);
            int max_index = std::max_element(compv.begin(), compv.end())-compv.begin();
            // std::cout<<"max_index:"<<max_index<<", ";
            if (1==max_index)
            {//车辆区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(0,0,255);
                inputSeg.at<uchar>(h, w) = 1;
            }
            else if (2==max_index)
            {//可行使区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(255,0,0);
                inputSeg.at<uchar>(h, w) = 2;
            }
            else if (3==max_index)
            {//行人区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(0,255,0);
                inputSeg.at<uchar>(h, w) = 3;
            }
            else if (4==max_index)
            {//行人区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(0,128,0);
                inputSeg.at<uchar>(h, w) = 4;
            }
            else if (5==max_index)
            {//行人区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(0,0,128);
                inputSeg.at<uchar>(h, w) = 5;
            }
            else if (6==max_index)
            {//行人区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(128,0,0);
                inputSeg.at<uchar>(h, w) = 6;
            }
        }
    }
    return true;
}


bool TrtAutoYolox::run(const cv::Mat ipt, std::vector<rmRECT> &DetectiontRects)
{
    // const int inputC = mTaskParams.inputDim[0];
    const int inputH = mTaskParams.inputDim[1];
    const int inputW = mTaskParams.inputDim[2];
    const int batchSize = mTaskParams.batchSize;

    if (1 != batchSize){
        cout <<"only batchsize=1 support"<<endl;
        return false;
    }
    
    cv::Mat resize_ipt;
    cv::resize(ipt, resize_ipt, cv::Size(inputW, inputH));

    if("GRAY" == mTaskParams.colorType){
        cv::cvtColor(resize_ipt,resize_ipt,cv::COLOR_BGR2GRAY);
    }else if("RGB" == mTaskParams.colorType){
        cv::cvtColor(resize_ipt,resize_ipt,cv::COLOR_BGR2RGB);
    }

    std::vector<TRT_TENSOR_ATTR_S> blobtensors(mTaskParams.batchSize*mTaskParams.outputTensorNames.size());
    float* iptarray = (float*)malloc(resize_ipt.channels()*sizeof(float)*resize_ipt.cols*resize_ipt.rows);

    if(!resize_ipt.isContinuous())
    {
        TRACE(LOG_ERROR,"%d: img should be continuos, got %s",__LINE__,resize_ipt.isContinuous());
        free(iptarray);
        return false;
    }

    for(int i=0; i<resize_ipt.cols*resize_ipt.rows*resize_ipt.channels(); i++)
    {
        *(iptarray+i) = (float)resize_ipt.data[i];
    }
    // for(int i=0;i<20;i++)
    // {
    //     cout<<iptarray[i]<<",";
    // }
    // cout<<endl;


    // for(int i=0;i<20;i++)
    // {
    //     cout<<(float)resize_ipt.data[i]<<",";
    // }
    // cout<<endl;

    bool ret = infer(iptarray, blobtensors);        // tensorrt7/trt_baseinfer.cpp

    free(iptarray);
    if(!ret)
    {
        TRACE(LOG_ERROR,"%d: infer failed",__LINE__);
        return false;
    }

    if(meshgrid.empty())
	{
		ret = GenerateMeshgrid();
	}
    const int maxinum_nms = 300;
	rmRECT res[maxinum_nms] = {0};
	int res_index = 0;

	int gridIndex = -2;

	float cx = 0, cy = 0, xf = 0, yf = 0;
	float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
	float ce_val = 0, cls_val = 0;
    
    for(int p=0; p<batchSize; p++)
    {

        for(int index = 0; index < headNum; index ++)
        {
            const float* cls = blobtensors[index*3+0].data;
            const float* reg = blobtensors[index*3+1].data;
            const float* ce = blobtensors[index*3+2].data;

            // std::cout<<"cls: "<<*cls<<std::endl;     //0x55b14e576f40
            // std::cout<<"objectness: "<<*ce<<std::endl;       //0x55b14e54df30
            // std::cout<<"bbox: "<<*reg<<std::endl;       //0x55b14e38bae0

            
            for(int h = 0; h < mapSize[index][0]; h++)
            {
                for(int w = 0; w < mapSize[index][1]; w++)
                {
                    gridIndex += 2;

                    ce_val = sigmoid(ce[h * mapSize[index][1] + w]);
                    // std::cout<<"ce_val:"<<ce_val<<std::endl;

                    for(int cl = 0; cl < class_num; cl ++)
                    {
                        cls_val = sigmoid(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * ce_val;
                        // std::cout<<"cls_val:"<<cls_val<<std::endl;

                        if(cls_val > objectThresh[cl])
                        {
                            cx = (meshgrid[gridIndex + 0] + reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] ) * strides[index];
                            cy = (meshgrid[gridIndex + 1] + reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index];
                            xf = exp(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] ) * strides[index];
                            yf = exp(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] ) * strides[index];

                            // std::cout<<"cx:"<<cx<<", "<<"cy:"<<cy<<", "<<"xf:"<<xf<<", "<<"yf:"<<yf<<std::endl;

                            xmin = cx - xf / 2;
                            ymin = cy - yf / 2;
                            xmax = cx + xf / 2;
                            ymax = cy + yf / 2;
                            
                            xmin = xmin > 0 ? xmin : 0;
                            ymin = ymin > 0 ? ymin : 0;
                            xmax = xmax < input_w ? xmax : input_w -1;
                            ymax = ymax < input_h ? ymax : input_h -1;
                            // std::cout<<"xmin:"<<xmin<<", "<<"ymin:"<<ymin<<", "<<"xmax:"<<xmax<<", "<<"ymax:"<<ymax<<std::endl;

                            if(xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                            {
                                rmRECT temp;
                                temp.left = int(xmin);
                                temp.top = int(ymin);
                                temp.right = int(xmax);
                                temp.bottom = int(ymax);
                                temp.classnum = cl;
                                temp.score = cls_val;

                                res[res_index++] = temp;
                                if(res_index >= maxinum_nms)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        // return true;
        
        int topK = res_index;
        int topKmin = (res_index < topK ? res_index : topK);
        if(res_index > 0)
        {				
            fcw_QuickSort(res, 0, res_index, pstStack, topKmin);
            
            for(int i = 0; i < topKmin; ++i)
            {
                int xmin1 = res[i].left;
                int ymin1 = res[i].top;
                int xmax1 = res[i].right;
                int ymax1 = res[i].bottom;
                int classId = res[i].classnum;
        
                if(res[i].classnum != -1)
                {
                    rmRECT RectTemp;
                    RectTemp.left = xmin1;
                    RectTemp.top = ymin1;
                    RectTemp.right = xmax1;
                    RectTemp.bottom = ymax1;
                    RectTemp.score = res[i].score;
                    RectTemp.classnum = classId + 1;

                    // std::cout<<"left:"<<xmin1<<", "<<"top:"<<ymin1<<", "<<"right:"<<xmax1<<", "<<"bottom:"<<ymax1<<std::endl;
                    
                    DetectiontRects.push_back(RectTemp);
        
                    for(int j = i + 1; j < topKmin; ++j)
                    {
                        //if(res[j].classnum == classId)
                        //{
                        int xmin2 = res[j].left;
                        int ymin2 = res[j].top;
                        int xmax2 = res[j].right;
                        int ymax2 = res[j].bottom;								
                        float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                        if(iou > nmsThresh)
                        {
                            res[j].classnum = -1;
                        }
                        //}
                    }
                }
            }
        }
    }
    
    return true;
}

bool TrtAutoYolox::run(const cv::Mat ipt, cv::Mat& resultSeg)
{
    // const int inputC = mTaskParams.inputDim[0];
    const int inputH = mTaskParams.inputDim[1];
    const int inputW = mTaskParams.inputDim[2];
    const int batchSize = mTaskParams.batchSize;

    if (1 != batchSize){
        TRACE(LOG_ERROR,"%d: only batchsize=1 support, got %d\n",__LINE__,batchSize);
        return false;
    }
    
    cv::Mat resize_ipt;
    cv::resize(ipt, resize_ipt, cv::Size(inputW, inputH));

    if("GRAY" == mTaskParams.colorType){
        cv::cvtColor(resize_ipt,resize_ipt,cv::COLOR_BGR2GRAY);
    }else if("RGB" == mTaskParams.colorType){
        cv::cvtColor(resize_ipt,resize_ipt,cv::COLOR_BGR2RGB);
    }

    std::vector<TRT_TENSOR_ATTR_S> blobtensors(mTaskParams.batchSize*mTaskParams.outputTensorNames.size());
    float* iptarray = (float*)malloc(resize_ipt.channels()*sizeof(float)*resize_ipt.cols*resize_ipt.rows);

    if(!resize_ipt.isContinuous())
    {
        TRACE(LOG_ERROR,"%d: img should be continuos, got %s\n",__LINE__,resize_ipt.isContinuous());
        free(iptarray);
        return false;
    }

    for(int i=0; i<resize_ipt.cols*resize_ipt.rows*resize_ipt.channels(); i++)
    {
        *(iptarray+i) = (float)resize_ipt.data[i];
    }


    bool ret = infer(iptarray, blobtensors);
    free(iptarray);
    if(!ret)
    {
        TRACE(LOG_ERROR,"%d: infer failed\n",__LINE__);
        return false;
    }

    int optH = 120;
    int optW = 208;
    for (int h = 0; h < optH; h++)
    {
        for (int w = 0; w < optW; w++)
        {
            std::vector<float> compv;
            compv.push_back(blobtensors[18].data[0*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[1*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[2*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[3*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[4*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[5*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[6*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[7*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[8*(optH*optW)+(h*optW+w)]);
            compv.push_back(blobtensors[18].data[9*(optH*optW)+(h*optW+w)]);
            std::vector<float>::iterator biggest = std::max_element(std::begin(compv),std::end(compv));
            int max_index = std::distance(std::begin(compv),biggest);
            if (4==max_index)
            {//车辆区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(0,0,255);
            }
            else if (5==max_index)
            {//可行使区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(255,0,0);
                // drawresult.at<cv::Vec3b>(h,w) = cv::Vec3b(0,0,255);
            }
            else if (9==max_index)
            {//行人区域
                resultSeg.at<cv::Vec3b>(h,w) = cv::Vec3b(0,255,0);
                // drawresult.at<cv::Vec3b>(h,w) = cv::Vec3b(0,0,255);
            }

        }
    }
    return true;
}
