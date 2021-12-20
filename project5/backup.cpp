#pragma GCC optimize(3)
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include<sys/time.h>
#include</opt/OpenBLAS/include/cblas.h>

#include "face_binary_cls.cpp"
#include "Matrix.hpp"
using namespace std;

using namespace cv;

const int ROW = 128;
const int COL = 128;
const int CHANNEL = 3;


//卷积函数
template <typename T>
void convolution(Matrix<T>& mat, const conv_param& conv, Matrix<T>& res_mat);

//矩阵乘法卷积运算
template <typename T>
void im2col_convolution(Matrix<T>& mat, const conv_param& conv, Matrix<T>& res_mat);

//Relu函数
template<typename T>
void Relu(Matrix<T>& in_mat);

//最大池化函数
template<typename T>
void max_pooling(Matrix<T>& in_mat, Matrix<T>& res_mat);

//全连接函数
template<typename T>
void full_connect(Matrix<T>& mat, fc_param& fc_params, float* vec, int size);

//softmax计算函数
void soft_max(float* vec, int size);


int main(){
    cout << fixed << setprecision(6);
    Mat img = imread("face.jpg");

    if(!img.data){//读入图片失败的检查
        cerr <<"Error in file " << __FILE__ <<" while loading image in line " << __LINE__ << " in FUCTION " << __FUNCTION__ << endl;
        exit(EXIT_FAILURE);
    }else{
        // cout << "Successfully loading image" << endl;
    }
   
    if(img.rows != ROW || img.cols != COL || img.channels() != CHANNEL){//确保矩阵规模正确
        cerr << "Size Mismatch of image in File " << __FILE__ << " in line " << __LINE__<< "in FUCTION " << __FUNCTION__ << endl;
        exit(EXIT_FAILURE);
    }

    Matrix<float> mat(ROW, COL, CHANNEL);//创建行数列数均为128，通道数为3的浮点数矩阵
    
    //在将读入的图片矩阵转化成实现过的矩阵类型
    //同时将BRG通道转换存储为RBG通道
    
    for(int r = 0; r < ROW; r++){
        for(int c = 0; c < COL; c++){
            mat(r, c, 2) = img.at<Vec3b>(r, c)[1] / 255.0;//规范化//g
            mat(r, c, 3) = img.at<Vec3b>(r, c)[0] / 255.0;//b
            mat(r, c, 1) = img.at<Vec3b>(r, c)[2] / 255.0;//r
        }
    }

      /*
    Matrix<float>test(5, 5, 3);
    for(int i = 0; i < 5; i++){
        test(0, i, 1) = 1;
        test(0, i, 2) = 1;
        test(0, i, 3) = 1;
    }
    for(int i = 0; i < 5; i++){
        test(1, i, 1) = 2;
        test(1, i, 2) = 2;
        test(1, i, 3) = 2;
    } 
    for(int i = 0; i < 5; i++){
        test(2, i, 1) = 3;
        test(2, i, 2) = 3;
        test(2, i, 3) = 3;
    }
    for(int i = 0; i < 5; i++){
        test(3, i, 1) = 2;
        test(3, i, 2) = 2;
        test(3, i, 3) = 2;
    }
    for(int i = 0; i < 5; i++){
        test(4, i, 1) = 1;
        test(4, i, 2) = 1;
        test(4, i, 3) = 1;
    }
    
//     typedef struct conv_param {
//     int pad;
//     int stride;
//     int kernel_size;
//     int in_channels;
//     int out_channels;
//     float* p_weight;
//     float* p_bias;
// } conv_param;
    float weight[27 * 3] = {1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1, 1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1, 1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1, 1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1, 1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1, 1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1, 1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1, 1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1, 1.0f, 1.0f, 1.0f, 2, 2, 2, 1, 1, 1};
    
    float bias[3] = {0, 0, 0};

    conv_param s = {1, 1, 3, 3, 3, weight, bias};
    Matrix<float> out(5, 5, 3);
    convolution(test, s, out);
    // cout << out << endl;


    return 0;
*/
   
    //128 128 3
    // printf("The height, width, channel of mat is %d, %d, %d\n", mat.getRow(), mat.getCol(), mat.getChannel());
    //根据公式算出第一层卷积后的结果矩阵的高度以及宽度，之后的规模换算同公式
    int conv_size1 = ((ROW - conv_params[0].kernel_size + 2 * conv_params[0].pad) / conv_params[0].stride + 1);
    
    //根据计算得到的规模以及通道数创建第一层卷积得到的矩阵
    Matrix<float> conv_mat1(conv_size1, conv_size1, conv_params[0].out_channels);
    
    //创建最大池化后的矩阵
    //此处+1再除以2的目的是考虑了矩阵长度或者宽度出现奇数的情况，使得最大池化操作不仅适用于偶数规模的矩阵,更具有普适性，之后的池化矩阵规模换算关系同理
    Matrix<float> max_pool_mat1((conv_size1 + 1) / 2, (conv_size1 + 1) / 2, conv_params[0].out_channels);
    // //64 64 16
    // printf("The height, width, channel of conv_mat1 is %d, %d, %d\n", conv_mat1.getRow(), conv_mat1.getCol(), conv_mat1.getChannel());
    // //32 32 16
    // printf("The height, width, channel of max_pool_mat1 is %d, %d, %d\n", max_pool_mat1.getRow(), max_pool_mat1.getCol(), max_pool_mat1.getChannel());

    timeval total_t1, total_t2;//计时
    gettimeofday(&total_t1, NULL);
    //进行第一层卷积运算
    im2col_convolution(mat, conv_params[0], conv_mat1);

    timeval relu_time1, relu_time2;
    double relu_time_use;
    gettimeofday(&relu_time1, NULL);
    Relu(conv_mat1);
    gettimeofday(&relu_time2, NULL);
    relu_time_use += (relu_time2.tv_sec - relu_time1.tv_sec) * 1000000.0 + (double)(relu_time2.tv_usec - relu_time1.tv_usec);
    
    // cout << conv_mat1 << endl;
    // return 0;
    max_pooling(conv_mat1, max_pool_mat1);//32 * 32 * 16
    // cout << max_pool_mat1 << endl;
    // return 0;
    

    //第一层卷积结束

    int conv_size2 = ((max_pool_mat1.getRow() - conv_params[1].kernel_size + 2 * conv_params[1].pad) / conv_params[1].stride + 1);
    Matrix<float> conv_mat2(conv_size2, conv_size2, conv_params[1].out_channels);
    Matrix<float> max_pool_mat2((conv_size2 + 1) / 2, (conv_size2 + 1) / 2, conv_params[1].out_channels);

    // //30 30 32
    // printf("The height, width, channel of conv_mat2 is %d, %d, %d\n", conv_mat2.getRow(), conv_mat2.getCol(), conv_mat2.getChannel());
    // //15 15 32
    // printf("The height, width, channel of max_pool_mat2 is %d, %d, %d\n", max_pool_mat2.getRow(), max_pool_mat2.getCol(), max_pool_mat2.getChannel());


    //第二层卷积运算
    im2col_convolution(max_pool_mat1, conv_params[1], conv_mat2);

    gettimeofday(&relu_time1, NULL);
    Relu(conv_mat2);
    gettimeofday(&relu_time2, NULL);
    relu_time_use += (relu_time2.tv_sec - relu_time1.tv_sec) * 1000000.0 + (double)(relu_time2.tv_usec - relu_time1.tv_usec);
    

    // cout << conv_mat2 << endl;
    // return 0;
    max_pooling(conv_mat2, max_pool_mat2);//15 * 15 * 32
    // cout << max_pool_mat2 << endl;
    // return 0;
   
    
    //第二层卷积结束

    int conv_size3 = ((max_pool_mat2.getRow() - conv_params[2].kernel_size + 2 * conv_params[2].pad) / conv_params[2].stride + 1);

    Matrix<float> conv_mat3(conv_size3, conv_size3, conv_params[2].out_channels);

    // //8 8 32
    // printf("The height, width, channel of conv_mat3 is %d, %d, %d\n", conv_mat3.getRow(), conv_mat3.getCol(), conv_mat3.getChannel());

    //第三层卷积运算
    im2col_convolution(max_pool_mat2, conv_params[2], conv_mat3);

    gettimeofday(&relu_time1, NULL);
    Relu(conv_mat3);
    gettimeofday(&relu_time2, NULL);
    relu_time_use += (relu_time2.tv_sec - relu_time1.tv_sec) * 1000000.0 + (double)(relu_time2.tv_usec - relu_time1.tv_usec);
    
    // cout << conv_mat3 << endl;
    // return 0;

    //第三层卷积结束

    //通过全连接层的out_feature计算结果向量的维数
    int vec_size = fc_params[0].out_features;
    //创建结果向量
    float* vec = new float[vec_size];
    memset(vec, 0, sizeof(float) * vec_size);

    //检查内存是否分配成功
    if(vec == nullptr){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "Failure in allocating memory." << endl;
        exit(EXIT_FAILURE);
    }
    

    //全连接层运算，将矩阵展平并转化成结果向量
    full_connect(conv_mat3, fc_params[0], vec, vec_size);

    //将结果向量转化成[0.0, 1.0]的概率区间范围
    soft_max(vec, vec_size);
    gettimeofday(&total_t2, NULL);
    double total_time_use = (total_t2.tv_sec - total_t1.tv_sec) * 1000 + (double)(total_t2.tv_usec - total_t1.tv_usec) / 1000.0;
    
    //打印最终运算结果
    printf("Background possibility: %f\nFace possibility: %f\n", vec[0], vec[1]);
    
    cout << "The total relu time used is " << relu_time_use << " us\n";
   
    cout << "The total time used in processing CNN using simple method is: " << total_time_use << "ms" << endl;

    return 0;
}

//朴素卷积运算
template <typename T>
void convolution(Matrix<T>& mat, const conv_param& conv, Matrix<T>& res_mat){
    const int mat_channel = mat.getChannel();
    const int res_mat_channel = res_mat.getChannel();
    //参数检查
    if(res_mat_channel != conv.out_channels){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The size of output matrix is not equal to the output channel number " << endl;
        exit(EXIT_FAILURE);
    }

    if(mat_channel != conv.in_channels){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "Mismatch of input channel number" << endl;
        exit(EXIT_FAILURE);
    }
    
    int in_channel = conv.in_channels;
    int out_channel = conv.out_channels;
    const int mat_r = mat.getRow();
    const int mat_c = mat.getCol();
    int ker_r = conv.kernel_size;
    int ker_c = conv.kernel_size;
    int padding = conv.pad;
    int stride = conv.stride;

    int res_height = res_mat.getRow();
    int res_width = res_mat.getCol();

    for(int out_ch = 0; out_ch < out_channel; out_ch++){//输出通道数量

        for(int in_ch = 0; in_ch < in_channel; in_ch++){//传入通道数量

            float ans = 0;
            // int conv_move = 0;
            
            for(int img_height = 0, res_h = 0; img_height + ker_r <= mat_r + 2 * padding; img_height += stride, res_h++){//图像高度 
                for(int img_width = 0, res_w = 0; img_width + ker_c <= mat_c + 2 * padding; img_width += stride, res_w++){//图像宽度 
                    
                    for(int ker_height = 0; ker_height < ker_r; ker_height ++){//核高度 
                        for(int ker_width = 0; ker_width < ker_c; ker_width ++){//核宽度 

                            if(img_height + ker_height >= padding && img_width + ker_width >= padding //在目标范围内
                                && img_height + ker_height < padding + mat_r && img_width + ker_width < padding + mat_c){
                               
                                ans += mat(img_height + ker_height - padding, img_width + ker_width - padding, in_ch + 1)
                                    * conv.p_weight[out_ch * (in_channel * 3 * 3) + in_ch * (3 * 3) + ker_height*ker_r + ker_width];
                                
                            }
                           
                        }
                    }
                    res_mat(res_h, res_w, out_ch + 1) += ans;
                    ans = 0;
                }
            }
           
        }
    }

    for(int ch = 1; ch <= out_channel; ch++){
        for(int h = 0; h < res_height; h++){
            for(int w = 0; w < res_width; w++){
                res_mat(h, w, ch) += conv.p_bias[ch - 1];
            }
        }
    }
    
}

/**/
//矩阵乘法卷积运算
template <typename T>
void im2col_convolution(Matrix<T>& mat, const conv_param& conv, Matrix<T>& res_mat){
    const int mat_channel = mat.getChannel();
    const int res_mat_channel = res_mat.getChannel();
    //参数检查
    if(res_mat_channel != conv.out_channels){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The size of output matrix is not equal to the output channel number " << endl;
        exit(EXIT_FAILURE);
    }

    if(mat_channel != conv.in_channels){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "Mismatch of input channel number" << endl;
        exit(EXIT_FAILURE);
    }
    if(mat.nums == nullptr){ 
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The input matrix's array is nullptr." << endl;
        exit(EXIT_FAILURE);
    }
    if(res_mat.nums == nullptr){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "the element array of the result matrix is nullptr" << endl;
        exit(EXIT_FAILURE);
    }
    
    int in_channel = conv.in_channels;
    int out_channel = conv.out_channels;
    const int mat_r = mat.getRow();
    const int mat_c = mat.getCol();
    int ker_size = conv.kernel_size * conv.kernel_size;
    int ker_r = conv.kernel_size;
    int ker_c = conv.kernel_size;
    int padding = conv.pad;
    int stride = conv.stride;


    int res_height = res_mat.getRow();
    int res_width = res_mat.getCol();
    
    Matrix<float> tmp_mat(ker_c * ker_r * in_channel, res_height * res_width, 1);
    Matrix<float> ker_mat(out_channel, in_channel * ker_size, 1);

    int mat_nums_size = mat.getCol() * mat.getRow() * mat.getChannel();
    int tmp_mat_size = tmp_mat.getCol() * tmp_mat.getRow() * tmp_mat.getChannel();
    int ker_mat_size = ker_mat.getCol() * ker_mat.getRow() * ker_mat.getChannel();

    //在实际访问之前先判断是否可以在即将进行的循环中保证访问不会越界导致程序崩溃
    if(((in_channel - 1) * ker_size + (ker_r - 1) * ker_c + ker_c - 1) * tmp_mat.getSpan() + res_height * res_width * tmp_mat.getChannel() > tmp_mat_size){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The size of array of the temporary matrix is not match of the source matrix." << endl;
        exit(EXIT_FAILURE);
    }
   
    for(int in_ch = 0; in_ch < in_channel; in_ch ++){//传入通道数量
        
        int span = in_ch * ker_size;//将最内层需要频繁计算的值先提前计算，提升性能
        
        int col_cnt = -1;
        for(int img_height = 0; img_height + ker_r <= mat_r + 2 * padding; img_height += stride){//图像高度 
            for(int img_width = 0; img_width + ker_c <= mat_c + 2 * padding; img_width += stride){//图像宽度 
                col_cnt ++;
                for(int ker_height = 0; ker_height < ker_r; ker_height ++){//核高度 
                    for(int ker_width = 0; ker_width < ker_c; ker_width ++){//核宽度 
                        
                        if(img_height + ker_height >= padding && img_width + ker_width >= padding //判断计算点是否位于目标范围内
                            && img_height + ker_height < padding + mat_r && img_width + ker_width < padding + mat_c){
                            
                            tmp_mat.nums[(span + ker_height * ker_c + ker_width) * tmp_mat.span + col_cnt * tmp_mat.channel] = 
                                mat.nums[(img_height + ker_height - padding) * mat.span + (img_width + ker_width - padding) * mat.channel + in_ch];//如果在范围内，就将对应位置赋值为输入矩阵对应的元素
                        
                        }else{
                            
                            tmp_mat.nums[(span + ker_height * ker_c + ker_width) * tmp_mat.span + col_cnt * tmp_mat.channel] = 0;//不在范围内就将其赋值为0
                        
                        }
                        
                    }
                }
              
                
            }
        }
    }

    //在实际访问之前先判断是否可以在即将进行的循环中保证访问不会越界导致程序崩溃

    if((out_channel - 1) * ker_mat.getSpan() + ((in_channel - 1) * ker_size + (ker_r - 1) *  ker_c + ker_c  - 1) * ker_mat.getChannel() > ker_mat_size){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The size of kernel transformation matrix is not equal to the kernel's size" << endl;
        exit(EXIT_FAILURE);
    }
   
    //将核转化成行矩阵ker_mat
    for(int o = 0; o < out_channel; o++){
        for(int i = 0; i < in_channel; i++){
            int span = i * ker_size;//将内层变量提前进行计算，提升性能
            int move = o*(in_channel*3*3) + i*(3*3);
            for(int ker_height = 0; ker_height < ker_r; ker_height++){
                for(int ker_width = 0; ker_width < ker_c; ker_width++){
                   
                    ker_mat.nums[o * ker_mat.span + (span + ker_height * ker_c + ker_width) * ker_mat.channel] = 
                    conv.p_weight[move + ker_height*ker_r + ker_width];
                }
            }
        }
    }
    
    Matrix<float> ans_mat = ker_mat * tmp_mat;
   
    for(int o = 0; o < out_channel; o++){
        for(int h = 0; h < res_height; h++){
            for(int c = 0; c < res_width; c++){
                
                res_mat.nums[h * res_mat.span + c * res_mat.channel + o] = ans_mat.nums[o * ans_mat.span  +  (h*res_width + c) * ans_mat.channel] + conv.p_bias[o];//将结果矩阵转化为输出矩阵并加上偏置
            }
        }
    }
}


//Relu激活函数
template<typename T>
void Relu(Matrix<T>& in_mat){
    
    const int height = in_mat.getRow();
    const int width = in_mat.getCol();
    const int channel = in_mat.getChannel();
    for(int h = 0; h < height; h++){
        for(int w = 0; w < width; w++){
            for(int ch = 1; ch <= channel; ch++){

                // in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1] = max(in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1], 0.0f);

                in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1] = (in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1] < 0)
                 ? 0 : in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1];
                
                // if(in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1] < 0){
                //     in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1] = 0;
                // }
                // in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1] *= (in_mat.nums[h * in_mat.span + w * in_mat.channel + ch - 1] > 0.f);
            
            }
        }
    }
}


//最大池化函数
template<typename T>
void max_pooling(Matrix<T>& in_mat, Matrix<T>& res_mat){
    
    const int src_height = in_mat.getRow();
    const int src_width = in_mat.getCol();
    const int src_channel = in_mat.getChannel();
    const int out_height = res_mat.getRow();
    const int out_width = res_mat.getCol();
    const int out_channel = res_mat.getChannel();
    
    //参数检查
    if((src_height % 2 == 0 && src_height != out_height * 2) || (src_width % 2 == 1 && src_height != out_height * 2 - 1)){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The height of output matrix: " << out_height << "should be half of the height of the input matrix: " << src_height << endl;
        exit(EXIT_FAILURE);
    }
    if((src_width % 2 == 0 && src_width != out_height * 2) || (src_width % 2 == 1 && src_width != out_width * 2 - 1)){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The width of output matrix: " << out_width << "should be half of the width of the input matrix: " << src_width << endl;
        exit(EXIT_FAILURE);
    } 
    if(src_channel != out_channel){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The channel of output matrix:" << out_channel << " should be the same as the output matrix: " << src_channel << endl;
        exit(EXIT_FAILURE);
    }

    for(int ch = 1; ch <= src_channel; ch++){
        for(int src_h = 0; src_h < src_height; src_h += 2){
            for(int src_c = 0; src_c < src_width; src_c += 2){
                // return nums[r * span + c * channel + ch - 1];
                float maximum = in_mat.nums[src_h * in_mat.span + src_c * in_mat.channel + ch - 1];
                for(int i = 0; i < 2; i++){
                    for(int j = 0; j < 2; j ++){
                        int h = src_h + i;
                        int c = src_c + j;
                        if(h < src_height && c < src_width && in_mat.nums[h * in_mat.span + c * in_mat.channel +ch - 1] > maximum){
                            maximum =in_mat.nums[h * in_mat.span + c * in_mat.channel +ch - 1];
                        }
                    }
                }
                res_mat.nums[((src_h + 1)/ 2) * res_mat.span + ((src_c + 1) / 2) * res_mat.channel + ch -1] = maximum;
            }
        }
    }

}

//全连接函数
template<typename T>
void full_connect(Matrix<T>& mat, fc_param& fc_params, float* vec, int size){
    const int channel = mat.getChannel();
    const int height = mat.getRow();
    const int width = mat.getCol();
    int out_feature = fc_params.out_features;
    int in_feature = fc_params.in_features;
    //参数检查
    if(vec == nullptr){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The input of the result vector is nullptr" << endl;
        exit(EXIT_FAILURE);
    }
    
    if(channel * height * width != in_feature){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The product of the channel and height and width of output matrix:" << channel * height * width << " should be the same as the in_feature " << in_feature << endl;
        exit(EXIT_FAILURE);
    }
    if(size != out_feature){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The size of output vector" << size << " should be the same as the out_feature " << out_feature << endl;
        exit(EXIT_FAILURE);
    }
    int cnt = 0;
    for(int o = 0; o < out_feature; o++){
        float ans = 0;
        
        for(int ch = 1; ch <= channel; ch++){
            for(int h = 0; h < height; h++){
                for(int c = 0; c < width; c++){
                   ans += mat.nums[h * mat.span + c * mat.channel +  ch - 1] * fc0_weight[cnt++];
                }
            }
        }
       
        vec[o] = ans + fc0_bias[o];
    }
    
}


void soft_max(float* vec, int size){
    //参数检查
    if(vec == nullptr){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The input of the result vector is nullptr" << endl;
        exit(EXIT_FAILURE);
    }
    if(size <= 0){
        cerr << "Error in file " << __FILE__ << " in function " << __FUNCTION__ << " in line " << __LINE__ << endl;
        cerr << "The input of vector size should be larger than 0" << endl;
        exit(EXIT_FAILURE);
    }

    float exp_sum = 0.0;
    
    for(int i = 0; i < size; i++){
        exp_sum += exp(vec[i]);
    }

    for(int i = 0; i < size; i++){
        vec[i] = exp(vec[i]) / exp_sum;
    }
    
}


