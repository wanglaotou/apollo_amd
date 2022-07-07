## 需要生成proto库
## 参数1：函数的源文件，参数2：函数名
load("//tools:python_rules.bzl", "py_proto_library")    # 手动加载
package(default_visibility = ["//visibility:public"])

proto_library(
    name = "student_proto",
    srcs = ["student.proto"],
)
# 生成C++相关的源文件
cc_proto_library(
    name = "student_cc",
    deps = [":student_proto"],
)
## 生成python的源文件
py_proto_library(
    name = "student_py",
    deps = [":student_proto"],
)

## 配置C++调用proto实现
# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary
cc_binary(
    name = "test_student",
    srcs = ["test_student.cc"],
    deps = [":student_cc"],
)


## ======================================================== ##
proto_library(
    name = "addints_proto",
    srcs = ["addints.proto"],
)
# 生成C++相关的源文件
cc_proto_library(
    name = "addints_cc",
    deps = [":addints_proto"],
)
## 生成python的源文件
py_proto_library(
    name = "addints_py",
    deps = [":addints_proto"],
)