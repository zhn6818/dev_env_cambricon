cmake_minimum_required(VERSION 3.10.2 )
project(CNRTVirgo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-O0 -g)

set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O0 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")
set(OpenCV_LIBS
    libopencv_core.so
    libopencv_core.so
    libopencv_imgproc.so
    libopencv_ximgproc.so
    libopencv_highgui.so
    libopencv_imgcodecs.so
    libopencv_videoio.so
    libopencv_video.so
    libopencv_optflow.so
    libopencv_calib3d.so
    libopencv_features2d.so
    libopencv_photo.so
    )


#包含头文件路径,include下若包含文件夹，则写作：include/。若是.h文件，则写作：include

message(${CMAKE_CURRENT_SOURCE_DIR})
message("~~~~~~~~~~~~~~~~")
include_directories(
    /usr/local/include
    /usr/local/neuware/include
    /usr/local/neuware/edge/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#添加动态库
link_directories(

    /usr/local/lib
    /usr/lib/x86_64-linux-gnu

    #寒武纪补充
    /usr/local/neuware/lib64
    /usr/local/neuware/samples
)


set(3RDLIBS
${OpenCV_LIBS}

stdc++ pthread libgflags.so

magicmind_runtime
cnrt
gflags
libcnrt.so
)

# add_subdirectory(src)
add_executable(infer includetest/pre_process.hpp includetest/utils.hpp srctest/infer_resnet50.cpp srctest/utils.cpp srctest/pre_process.cpp)
target_link_libraries(
    infer
    ${3RDLIBS}
)


add_subdirectory(src)
add_subdirectory(test)