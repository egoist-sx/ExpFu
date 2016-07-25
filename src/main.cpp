#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void ExposureFusion(const vector<Mat> &images, Mat &out, float wcon = 1.f, float wsat = 1.f, float wexp = 1.f) {
    Size size = images[0].size();
    vector<Mat> weights(images.size());
    Mat weight_sum = Mat::zeros(size, CV_32F);
    
    for (size_t i = 0; i < images.size(); i++) {
        Mat img, gray, contrast, saturation, wellexp;
        vector<Mat> splitted(3);
        
        images[i].convertTo(img, CV_32F, 1.f/255.f);
        cvtColor(img, gray, COLOR_RGB2GRAY);
        split(img, splitted);
        
        Laplacian(gray, contrast, CV_32F);
        contrast = abs(contrast);
        
        Mat mean = Mat::zeros(size, CV_32F);
        for (int c = 0; c < 3; c++) {
            mean += splitted[c];
        }
        mean /= 3;
        
        saturation = Mat::zeros(size, CV_32F);
        for (int c = 0; c < 3; c++) {
            Mat deviation = splitted[c] - mean;
            pow(deviation, 2.f, deviation);
            saturation += deviation;
        }
        sqrt(saturation, saturation);
        
        wellexp = Mat::ones(size, CV_32F);
        for (int c = 0; c < 3; c++) {
            Mat exposure = splitted[c] - 0.5f;
            pow(exposure, 2.0f, exposure);
            exposure = -exposure/0.08f;
            exp(exposure, exposure);
            wellexp = wellexp.mul(exposure);
        }
        
        pow(contrast, wcon, contrast);
        pow(saturation, wsat, saturation);
        pow(wellexp, wexp, wellexp);
        
        weights[i] = Mat::ones(size, CV_32F);
        if (wcon > 0) 
            weights[i] = weights[i].mul(contrast);
        if (wsat > 0)
            weights[i] = weights[i].mul(saturation);
        if (wexp > 0)
            weights[i] = weights[i].mul(wellexp);
        weights[i] = weights[i] + 1e-12f;
        weight_sum += weights[i];
    }
    
    int maxlevel = static_cast<int>(logf(static_cast<float>(min(size.width, size.height)))/logf(2.f));
    maxlevel -= 1;
    vector<Mat> res_pyr(maxlevel + 1);
    
    for (size_t i = 0; i <images.size(); i++) {
        weights[i] /= weight_sum;
        Mat img;
        images[i].convertTo(img, CV_32F, 1.0/255.f);
        
        vector<Mat> img_pyr, weight_pyr;
        buildPyramid(img, img_pyr, maxlevel);
        buildPyramid(weights[i], weight_pyr, maxlevel);
        
        // Laplacian pyramid
        for (int lvl = 0; lvl < maxlevel; lvl++) {
            Mat up;
            pyrUp(img_pyr[lvl + 1], up, img_pyr[lvl].size());
            img_pyr[lvl] -= up;
            //displayMat(img_pyr[lvl]);
        }
        
        for (int lvl = 0; lvl <= maxlevel; lvl++) {
            vector<Mat> splitted(3);
            split(img_pyr[lvl], splitted);
            for (int c = 0; c < 3; c++) {
                splitted[c] = splitted[c].mul(weight_pyr[lvl]);
            }
            merge(splitted, img_pyr[lvl]);
            if (res_pyr[lvl].empty()) {
                res_pyr[lvl] = img_pyr[lvl];
            } else {
                res_pyr[lvl] += img_pyr[lvl];
            }
        }
    }
    
    for (int lvl = maxlevel; lvl > 0; lvl--) {
        Mat up;
        pyrUp(res_pyr[lvl], up, res_pyr[lvl-1].size());
        res_pyr[lvl-1] += up;
    }
    
    out.create(size, CV_32FC3);
    res_pyr[0].copyTo(out);
}

void addCircularPadding(const Mat &src, Mat &des) {
    int row = src.rows; int col = src.cols;
    int padding = row;
    des = Mat(row, col+padding*2, src.type());
    src.copyTo(des(Rect_<int>(padding, 0, col, row)));
    des(Rect_<int>(col, 0, padding, row)).copyTo(des(Rect_<int>(0, 0, padding, row))) ;
    des(Rect_<int>(padding, 0, padding, row)).copyTo(des(Rect_<int>(col+padding, 0, padding, row)));
}

int main(int argc, char **argv) {
    bool usePadding = false;
    if (argc == 5) {
        cout << "normal 3 HDR" << endl;
    } else if (argc == 6) {
        cout << "HDR for 2:1 panorama image" << endl;
        usePadding = true;
    } else if (argc == 7) {
        cout << "normal 5 HDR" << endl;
    } else {
        cout << "Required 6 params but not given exactly 6" << endl; 
        return 1;
    }
    
    vector<Mat> src;
    for (int i = 0; i < 5; i++) {
        Mat img = imread(argv[i+1]);
        if (usePadding) {
            Mat pad;
            addCircularPadding(img, pad);
            src.push_back(pad);
        } else 
            src.push_back(img);
    }
    Mat fusion;
    
    ExposureFusion(src, fusion);
    if (usePadding)
        imwrite(argv[4], fusion(Rect_<int>(fusion.rows, 0, fusion.rows*2, fusion.rows))*255);
    else {
        std::cout << "writing to " << argv[6] << std::endl;
        imwrite(argv[6], fusion*255);
    }
    return 0;
}
