/*
 * @Author: your name
 * @Date: 2021-09-16 10:42:34
 * @LastEditTime: 2021-09-18 14:07:34
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /WEBAI_SERVER/include/commons/log.h
 */
#ifndef _LOG_H
#define _LOG_H

#include <vector>
#include <string>
#include <map>

#define LOG_FAULT   (1 << 0)
#define LOG_ERROR   (1 << 1)
#define LOG_WARN    (1 << 2)
#define LOG_INFO    (1 << 3)
#define LOG_DEBUG   (1 << 4)

#define global_trace 0x1F //全部打印
// #define global_trace  0xF //仅打印及INFO以上
// #define global_trace  0x7 //仅打印及WARN以上
// #define global_trace  0x3 //仅打印及ERROR以上

/*LOG PRINTF*/
#define TRACE(trace,fmt, ...)                                               \
    do {                                                                \
        if ((global_trace & (trace)) == LOG_FAULT)                      \
            printf("[FAULT] <%s> " fmt "", __FUNCTION__, ##__VA_ARGS__);          \
        else if ((global_trace & (trace)) == LOG_ERROR)                 \
            printf("[ERROR] <%s> " fmt "", __FUNCTION__, ##__VA_ARGS__);          \
        else if ((global_trace & (trace)) == LOG_WARN)                  \
            printf("[WARN]  <%s> " fmt "", __FUNCTION__, ##__VA_ARGS__);          \
        else if ((global_trace & (trace)) == LOG_INFO)                  \
            printf("[INFO]  <%s> " fmt "", __FUNCTION__, ##__VA_ARGS__);          \
        else if ((global_trace & (trace)) == LOG_DEBUG)                 \
            printf("[DEBUG] <%s> " fmt "", __FUNCTION__, ##__VA_ARGS__);          \
    } while(0)  

#endif //_LOG_H