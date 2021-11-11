cmake_minimum_required(VERSION 3.8.2)
PROJECT (PROJECT)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)
add_definitions(-DDEBUG -DRKAIQ)



aux_source_directory(${PROJECT_SOURCE_DIR}/common COMMON_SRCS)
set(LIBS easymedia rkaiq pthread rknn_api rtsp)
set(OpenCV_LIBS 
	opencv_core
	opencv_imgcodecs
	opencv_imgproc
  opencv_videoio)
#rkmedia
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/easymedia)
include_directories(${PROJECT_SOURCE_DIR}/include/rkmedia)
# rkaiq
include_directories(${CMAKE_SYSROOT}/usr/include/rkaiq/algos)
include_directories(${CMAKE_SYSROOT}/usr/include/rkaiq/common)
include_directories(${CMAKE_SYSROOT}/usr/include/rkaiq/iq_parser)
include_directories(${CMAKE_SYSROOT}/usr/include/rkaiq/uAPI)
include_directories(${CMAKE_SYSROOT}/usr/include/rkaiq/xcore)

link_directories(${PROJECT_SOURCE_DIR}/lib
				 ${PROJECT_SOURCE_DIR}/lib/opencv)

add_executable(blazeface main.cpp BlazeFace.cpp BlazeFace.h ${COMMON_SRCS})
target_link_libraries(blazeface ${LIBS} ${OpenCV_LIBS})

install(TARGETS blazeface DESTINATION bin)

