#include "BlazeFace.h"
#include <iostream>
#include <chrono>
#include <thread>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <sys/time.h>
#include <thread>

using namespace std;


static double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


int main(int argc, char **argv){
    cv::Mat image;
    int count = 0;
    int fps;
    double first_time,last_time;
    BlazeFace face(argv[1]);
    face.model_init();

    std::string arg2(argv[2]);
    image = cv::imread(arg2, cv::IMREAD_COLOR);

    first_time = what_time_is_it_now();

    detect_result_group_t detect_result_group;
    face.forward(image.clone(), &detect_result_group);
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        cv::rectangle(image, cv::Rect(det_result->box.left, det_result->box.top, det_result->box.right-det_result->box.left, det_result->box.bottom-det_result->box.top), cv::Scalar(0, 255, 0));
        for(int j = 0; j < 6; j++){
            cv::circle(image, cv::Point(det_result->keypoints[j][0],det_result->keypoints[j][1]),3, cv::Scalar(255,0,0),CV_FILLED, 2,0);
        }
    }
    last_time = what_time_is_it_now();
    fps = (int)1/(last_time - first_time);
    std::cout << "fps: "<< fps  << std::endl;
    cv::imwrite("output.jpg", image);

    std::cout << "exit demo end" << std::endl;


    return 0;
}