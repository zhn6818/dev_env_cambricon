cmake_minimum_required(VERSION 3.5.1)

project(test_detect)


#yolov7 detect
add_executable(test_detect test_detect.cpp)
target_link_libraries(
    test_detect
    CNRTVirgo
    ${3RDLIBS}
)