load("@rules_cc//cc:defs.bzl", "cc_library")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "yolox",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//cyber",
        "@opencv//:core",
        "@opencv//:imgcodecs",
        "@opencv//:imgproc",
        "@opencv//:highgui",
        "@local_config_cuda//cuda:cudnn_header",
        "@local_config_cuda//cuda:cudart",
        "@local_config_tensorrt//:tensorrt",
    ],
)

cpplint()
