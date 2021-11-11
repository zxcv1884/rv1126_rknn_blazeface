#ifndef _BLAZEFACE_H_
#define _BLAZEFACE_H_

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ctime>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "rknn/rknn_api.h"

#define OBJ_NUMB_MAX_SIZE 64

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    int keypoints[6][2];
    BOX_RECT box;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;



class BlazeFace {
    typedef cv::Mat Mat;

private:
    unsigned char *m_model;
    int m_model_size;
    rknn_context m_context;
    rknn_input_output_num m_io_num;
    rknn_output* m_outputs;
    rknn_tensor_attr* output_attrs;
    float scoreThreshold = 0.7;
    float sigmoidScoreThreshold = logf(scoreThreshold/(1-scoreThreshold));
    float iouThreshold = 0.3;
    int inputHeight = 128;
    int inputWidth = 128;
    int channels = 3;
    std::vector<double> anchor;
    struct str
    {
        float value;
        int index;
    };

public:
    explicit BlazeFace(const char *filename);

    static void print_tensor(rknn_tensor_attr *attr) {
        printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
               attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
               attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
    }

    bool model_init();

    void compute_iou(std::vector<std::vector<float>> boxes, std::vector<unsigned int> scores_indexes, std::vector<float> areas, int index, std::vector<float> &ious);

    std::vector<unsigned int> non_max_suppression(std::vector<std::vector<float>> boxes, std::vector<float> scores_vector, float threshold, int size);
    

    bool forward(Mat img, detect_result_group_t* detect_result_group);

    template<typename T>
    std::vector<unsigned int> argsort(const std::vector<T> array);

    void postprocess(detect_result_group_t *group, int w, int h);

    ~BlazeFace() {
        delete m_model;
        // Release
        if (m_context >= 0) {
            rknn_destroy(m_context);
        }

        // Release rknn_outputs
        rknn_outputs_release(m_context, m_io_num.n_output, m_outputs);
    };
};

#endif
