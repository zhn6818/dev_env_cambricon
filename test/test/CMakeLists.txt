cmake_minimum_required(VERSION 3.5.1)

project(test_classify)

add_executable(test test.cpp )
# set_target_properties(dvpp_test  PROPERTIES LINK_FLAGS "-Wl,--disable-new-dtags,-rpath,/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/acllib/lib64")
target_link_libraries(
    test
    CNRTVirgo
    ${3RDLIBS}
)