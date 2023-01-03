cmake_minimum_required(VERSION 3.10.2)

project(test)
add_compile_options(-std=c++11)
add_definitions(-O0 -g)
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_CURRENT_LIST_DIR}/../install/include/include/
)
link_directories(
    ${CMAKE_CURRENT_LIST_DIR}/../install/lib/
    /data1/3trirdparty/acllib/lib64/stub
    
)
# message(${CMAKE_CURRENT_LIST_DIR}/../install/lib/)

add_subdirectory(test)