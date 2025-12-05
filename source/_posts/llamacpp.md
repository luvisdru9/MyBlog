---
title: llama.cpp编译过程中的cmake版本问题
author: luvisdru9
date: 2025-07-29 23:00:00
updated: 2025-07-29 23:00:00
tags: 

  - cmake
  - orin nx

categories: Solution
description: llama.cpp编译过程中的cmake版本问题
keywords:

  - cmake
  - orin nx

#top_img:
#comments:
#cover: img.png
#toc:
#toc_number:
#toc_style_simple:
#copyright:
#copyright_author:
#copyright_author_href:
#copyright_url:
#copyright_info:
#mathjax:
#katex:
#aplayer:
#highlight_shrink:
#aside:
#abcjs:
#noticeOutdate:
---

## cmake版本问题
今天在Orin NX上进行GPU版本的llama.cpp编译时，遇到了下述问题：

```
cmake -B build -DGGML_CUDA=ON
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- GGML_SYSTEM_ARCH: ARM
-- Including CPU backend
-- Found OpenMP_C: -fopenmp  
-- Found OpenMP_CXX: -fopenmp  
-- Found OpenMP: TRUE   
-- ARM detected
-- ARM -mcpu not found, -mcpu=native will be used
-- ARM feature FMA enabled
-- Adding CPU backend variant ggml-cpu: -mcpu=native 
CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:1 (cmake_minimum_required):
  CMake 3.18 or higher is required.  You are running version 3.16.3


-- Configuring incomplete, errors occurred!
See also "/home/amov/Desktop/llama.cpp/build/CMakeFiles/CMakeOutput.log".
See also "/home/amov/Desktop/llama.cpp/build/CMakeFiles/CMakeError.log".
```

具体原因是cmake版本过低，需要升级至``>=3.18``版本，具体流程如下：

1. 首先在官网下载对应的二进制文件，我需要安装4.0.0版本，由于Orin NX为arm架构，这里我选择aarch64后缀的脚本：``cmake-4.0.0-linux-aarch64.sh``
2. 下载下来后，使用``chmod +x cmake-4.0.0-linux-aarch64.sh``给予可执行权限，然后使``sudo sh cmake-4.0.0-linux-aarch64.sh``进行解压（这里需要注意安装时的路径提示）
3. 记住安装路径后，``export PATH=/"your cmake path"/bin:$PATH``，我这里是``export PATH=/home/amov/Desktop/cmake-4.0.0-linux-aarch64/bin:$PATH``
4. 而后执行以便永久生效：

```
echo 'export PATH=/opt/cmake/bin:$PATH' >>~/.bashrc
source ~/.bashrc
```
5. 最后``cmake --version``检查是否升级成功


## 参考：https://blog.csdn.net/u013172930/article/details/148203298