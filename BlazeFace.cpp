#include "BlazeFace.h"
#include <memory.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
using namespace std;

BlazeFace::BlazeFace(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr) {
        printf("fopen %s fail!\n", filename);
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    m_model = (unsigned char *) malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(m_model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(m_model);
    }
    m_model_size = model_len;
    if (fp) {
        fclose(fp);
    }
    printf("read model file succeed\n");
}

bool BlazeFace::model_init() {
    
    std::ifstream fin("./anchors.bin", std::ios::binary);

    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(double);
    fin.seekg(0, std::ios::beg);
    anchor.resize(num_elements);
    fin.read(reinterpret_cast<char*>(&anchor[0]), num_elements*sizeof(double));

    int ret = rknn_init(&m_context, m_model, m_model_size, 0);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return false;
    }
    // Get Model Input Output Info
    ret = rknn_query(m_context, RKNN_QUERY_IN_OUT_NUM, &m_io_num, sizeof(m_io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return false;
    }
    printf("model input num: %d, output num: %d\n", m_io_num.n_input, m_io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[m_io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < m_io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(m_context, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return false;
        }
        print_tensor(&(input_attrs[i]));
    }

    m_outputs = (rknn_output *)malloc(sizeof(rknn_output) * m_io_num.n_output);
    memset(m_outputs, 0, sizeof(rknn_output) * m_io_num.n_output);
    for (size_t i = 0; i < m_io_num.n_output; i++) {
        m_outputs[i].want_float = 1;
        // m_outputs[i].is_prealloc = false;
    }
    std::cout << "m_io_num.n_output: " << m_io_num.n_output << endl;

    printf("output tensors:\n");
    // rknn_tensor_attr output_attrs[m_io_num.n_output];

    output_attrs =
        (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * m_io_num.n_output);
    memset(output_attrs, 0, sizeof(rknn_tensor_attr) * m_io_num.n_output);

    for (size_t i = 0; i < m_io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(m_context, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return false;
        }
        print_tensor(&(output_attrs[i]));
    }

    return true;
}

template<typename T>
std::vector<unsigned int> BlazeFace::argsort(const std::vector<T> array) {
    std::vector<unsigned int> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
            [array](int left, int right) -> bool {
                // sort indices according to corresponding array element
                return array[left] < array[right];
            });

    return indices;
}

void BlazeFace::compute_iou(std::vector<std::vector<float>> boxes, std::vector<unsigned int> scores_indexes, std::vector<float> areas, int index, std::vector<float> &ious){

    std::vector<float> boxes_area(scores_indexes.size());
    std::vector<float> iouys1(scores_indexes.size());
    std::vector<float> iouxs1(scores_indexes.size()); 
    std::vector<float> iouys2(scores_indexes.size());
    std::vector<float> iouxs2(scores_indexes.size());

    float box_area = areas[index];
    std::vector<float> unions(scores_indexes.size());

    std::vector<float> intersections(scores_indexes.size());
    std::vector<float> preintersections1(scores_indexes.size(), 0);
    std::vector<float> preintersections2(scores_indexes.size(), 0);

    for(int j = 0; j < scores_indexes.size(); j++){
        iouys1[j] = boxes[index][0];
        iouxs1[j] = boxes[index][1]; 
        iouys2[j] = boxes[index][2];
        iouxs2[j] = boxes[index][3];
    }
 
    for(int j = 0; j < scores_indexes.size(); j++){
        if (iouys1[j] < boxes[scores_indexes[j]][0]) iouys1[j] = boxes[scores_indexes[j]][0];
        if (iouxs1[j] < boxes[scores_indexes[j]][1]) iouxs1[j] = boxes[scores_indexes[j]][1];
        if (iouys2[j] > boxes[scores_indexes[j]][2]) iouys2[j] = boxes[scores_indexes[j]][2];
        if (iouxs2[j] > boxes[scores_indexes[j]][3]) iouxs2[j] = boxes[scores_indexes[j]][3];
        boxes_area[j] = areas[scores_indexes[j]];
    }

    for(int j = 0; j < scores_indexes.size(); j++){
        if (preintersections1[j] < iouys2[j] - iouys1[j]) preintersections1[j] = iouys2[j] - iouys1[j];
        if (preintersections2[j] < iouxs2[j] - iouxs1[j]) preintersections2[j] = iouxs2[j] - iouxs1[j];
    }

    for(int j = 0; j < scores_indexes.size(); j++){
        intersections[j] = preintersections1[j] * preintersections2[j];
    }

    for(int j = 0; j < scores_indexes.size(); j++){
        unions[j] = box_area + boxes_area[j] - intersections[j];
    }

    for(int j = 0; j < scores_indexes.size(); j++){
        ious[j] = intersections[j] / unions[j];
    }
}

std::vector<unsigned int> BlazeFace::non_max_suppression(std::vector<std::vector<float>> boxes, std::vector<float> scores_vector, float threshold, int size) {
    std::vector<float> ys1(size);
    std::vector<float> xs1(size);
    std::vector<float> ys2(size);
    std::vector<float> xs2(size);
    std::vector<float> areas(size);
    int index = 0;

    for(int i = 0; i < size; i++){
        ys1[i] = boxes[i][0];
        xs1[i] = boxes[i][1];
        ys2[i] = boxes[i][2];
        xs2[i] = boxes[i][3];
        areas[i] = (ys2[i] - ys1[i]) * (xs2[i] - xs1[i]);
    }

    //std::vector<float> scores_vector {scores, scores + size};
    std::vector<unsigned int> scores_indexes = argsort(scores_vector);
    std::vector<unsigned int> boxes_keep_index;

    while(scores_indexes.size() > 0){
        index = scores_indexes.back();
        scores_indexes.pop_back();
        boxes_keep_index.push_back(index);
        if (scores_indexes.size() == 0){
            break;
        }

        std::vector<float> ious(scores_indexes.size());
        compute_iou(boxes, scores_indexes, areas, index, ious);
        std::vector<float> filtered_indexes;
        for(int j = 0; j < scores_indexes.size(); j++){
            if(ious[j] > iouThreshold){
                filtered_indexes.push_back(j);
            }
        }

        std::vector<unsigned int> scores_indexes_temp;
        for (int j = 0; j <scores_indexes.size(); j++){
            bool isIn = true;
            for (int z = 0; z < filtered_indexes.size(); z++){
                if(j == filtered_indexes[z]) isIn = false;
            }
            if(isIn) scores_indexes_temp.push_back(scores_indexes[j]);
        }
        
        scores_indexes = scores_indexes_temp;
    }
    return boxes_keep_index;
}

void BlazeFace::postprocess(detect_result_group_t *group, int w, int h){
    group->count = 0;
    float *output0 = (float *)m_outputs[0].buf;
    float *output1 = (float *)m_outputs[1].buf;
    int goodDetect[896];
    int count = 0;
    for (int i = 0; i < 896; i++){
        if(output1[i] > sigmoidScoreThreshold){
            goodDetect[count] = i;
            count++;
        }
    }

    std::vector<float> scores(count);

    for(int i = 0; i < count; i++){
        scores[i] = 1./(1.+exp(-output1[goodDetect[i]]));
    }
    
    std::vector<std::vector<float>> boxes(count, std::vector<float>(4));
    std::vector<std::vector<std::vector<float>>> keypoints(count, std::vector<std::vector<float>>(6, std::vector<float>(2)));

    for(int i = 0; i < count; i++){
        int detectionIdx = goodDetect[i];
        float sx = output0[detectionIdx*16];
		float sy = output0[detectionIdx*16+1];
		float w = output0[detectionIdx*16+2];
		float h = output0[detectionIdx*16+3];
        float cx = sx + anchor[detectionIdx * 4] * inputWidth;
		float cy = sy + anchor[detectionIdx * 4 + 1] * inputHeight;

		cx /= inputWidth;
		cy /= inputHeight;
		w /= inputWidth;
		h /= inputHeight;
        for(int j = 0; j < 6; j++){
            float lx = output0[detectionIdx*16 + (4 + (2 * j) + 0)];
			float ly = output0[detectionIdx*16 + (4 + (2 * j) + 1)];
			lx += anchor[detectionIdx * 4] * inputWidth;
			ly += anchor[detectionIdx * 4 + 1] * inputHeight;
			lx /= inputWidth;
		    ly /= inputHeight;
            keypoints[i][j][0] = lx;
            keypoints[i][j][1] = ly;
        }
        boxes[i][0] = cx - w * 0.5;
        boxes[i][1] = cy - h * 0.5;
        boxes[i][2] = cx + w * 0.5;
        boxes[i][3] = cy + h * 0.5;
    }
    std::vector<unsigned int> boxes_keep_index = non_max_suppression(boxes, scores, iouThreshold, count);

    int bestIndex = 0;
    float biggestRect = 0;
    if(boxes_keep_index.size() > 0){
        std::vector<std::vector<std::vector<float>>> filtered_keypoints(boxes_keep_index.size(), std::vector<std::vector<float>>(6, std::vector<float>(2)));
        std::vector<std::vector<float>> filtered_boxes(boxes_keep_index.size(), std::vector<float>(4));
        std::vector<float> filtered_scores(boxes_keep_index.size());
        for(int i = 0; i < boxes_keep_index.size(); i++){
            for(int j = 0; j < 6; j++){
                for(int z = 0; z <2; z++){
                    filtered_keypoints[i][j][z] = keypoints[boxes_keep_index[i]][j][z];
                }
            }
            for(int j = 0; j < 4; j++){
                filtered_boxes[i][j] = boxes[boxes_keep_index[i]][j];
            }
            filtered_scores[i] = scores[boxes_keep_index[i]];
        }
        group->count = boxes_keep_index.size();
        for(int i = 0; i < boxes_keep_index.size(); i++){
            group->results[i].box.left = (int)(filtered_boxes[i][0] * w);
            group->results[i].box.top = (int)(filtered_boxes[i][1] * h);
            group->results[i].box.right = (int)(filtered_boxes[i][2] * w);
            group->results[i].box.bottom = (int)(filtered_boxes[i][3] * h);
            for(int j = 0; j < 6; j++){
                group->results[i].keypoints[j][0] = (int)(filtered_keypoints[i][j][0] * w);
                group->results[i].keypoints[j][1] = (int)(filtered_keypoints[i][j][1] * h);
            }
        }
    }
}

bool BlazeFace::forward(Mat img, detect_result_group_t* detect_result_group) {
    
    int w = img.cols;
    int h = img.rows;
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(128, 128));

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img.cols * img.rows * img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;

    int ret = rknn_inputs_set(m_context, m_io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return false;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(m_context, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return false;
    }

    // Get Output
    ret = rknn_outputs_get(m_context, m_io_num.n_output, m_outputs, NULL);
    postprocess(detect_result_group, w, h);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return false;
    }
    return true;

}
