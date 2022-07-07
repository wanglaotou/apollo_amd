## 方式一
# package(default_visibility = ["//visibility:public"])
# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library
cc_library(
    name = "hello_great_lib",
    srcs = ["hello_great.cc"],
    hdrs = ["hello_great.h"],
    ## 方式二
    visibility = ["//demo_main:__pkg__"]
)

# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary
cc_binary(
    name = "hello_world",
    srcs = ["hello_world.cc"],
    copts = [],
    deps = [":hello_great_lib"],
)