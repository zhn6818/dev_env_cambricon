cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-std=c++11)
add_definitions(-O0 -g)
file(
    GLOB_RECURSE
    SOURCES_FILES
    classify/*.cpp
    Util/*.cpp
    ModelProcess/*.cpp
)
file(
    GLOB_RECURSE
    HEADER_FILES
    ${CMAKE_CURRENT_LIST_DIR}/../include/*.h
    ${CMAKE_CURRENT_LIST_DIR}/Util/*.h
    ${CMAKE_CURRENT_LIST_DIR}/ModelProcess/*.h
    # /data1/code/Ascend_virgo/src/Utils/*.h
)
message(${CMAKE_CURRENT_LIST_DIR})
message(${HEADER_FILES})
message("~~~~~~~~~~~~~~~")
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/../include/
    ${CMAKE_CURRENT_LIST_DIR}/Util
    ${CMAKE_CURRENT_LIST_DIR}/ModelProcess
)

add_library(
    ${PROJECT_NAME} SHARED
    ${SOURCES_FILES}
    ${HEADER_FILES}
)
target_link_libraries(${PROJECT_NAME} ${3RDLIBS})

install(TARGETS ${PROJECT_NAME}
DESTINATION lib)
install(DIRECTORY ../include
DESTINATION include)
