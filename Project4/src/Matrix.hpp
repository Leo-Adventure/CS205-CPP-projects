#ifndef _TEST_MATRIX_H
#define _TEST_MATRIX_H
#include<iostream>
#include<cstring>
#include<fstream>
#include<sstream>


template<class T>
class Matrix{
private:
    int row;
    int col;
    int channel;
    T *nums;
    int ROIBeginRow;
    int ROIBeginCol;
    int ROIRow;
    int ROICol;
    int span;
    static int num_matrices;
    int* ref_count;
public:

    //返回矩阵对象存在个数
    static int Matrix_num();
    //返回共用该元素数组矩阵对象个数
    int get_ref_count(){
        return ref_count[0];
    }

    int getRow(){
        return row;
    }
    int getCol(){
        return col;
    }
    int getChannel(){
        return channel;
    }
    int getSpan(){
        return span;
    }

    //构造器
    //根据传入数组来创建矩阵的构造函数
    Matrix(int row = 3, int col = 3, int channel = 1, T* nums = nullptr, int ROIBeginRow = 0, int ROIBeginCol = 0, int ROIRow = 0, int ROICol = 0, int *ref_count = nullptr, int span = 0);
    //根据传入文件流来创建矩阵的构造函数
    Matrix(int row, int col, int channel, std::ifstream & fin, int size);
    
    //向量点乘
    static T dotmul(int r1, T* nums1, int r2, T* nums2);

    //复制构造器
    Matrix(const Matrix&);
    //析构函数
    ~Matrix();

    //矩阵加法运算符
    Matrix operator+(T num) const;
    Matrix operator+(const Matrix &mat)const ;
    Matrix& operator+=(const Matrix &mat) ;
    Matrix& operator+=(T num);
    //友元函数
    template<class TT>
    friend Matrix operator+(TT num, const Matrix<TT>& mat){
        return mat + num;
    }

    //矩阵减法运算符
    Matrix operator-(T num) const ;
    Matrix operator-(const Matrix &mat)const ; 
    
    Matrix& operator-=(T num);
    Matrix& operator-=(const Matrix &mat);

    template<class TT>
    friend Matrix operator-(TT num, const Matrix<TT>& mat){
        return mat - num;
    }
    
    //矩阵除法运算符
    Matrix operator/(T num) const ;
    Matrix& operator/=(T num);

    //矩阵乘法运算符
    Matrix operator*(T num) const;
    Matrix operator*(Matrix &mat);
    Matrix& operator*=(T num);
    
    template<class TT>
    friend Matrix operator*(TT num, const Matrix<TT>& mat){
        return mat * num;
    }

    //矩阵转置运算符
    Matrix operator!() const;
    
    //自增运算符
    //前缀
    Matrix& operator++();
    //后缀
    Matrix operator++(int);
    //自减运算符
    //前缀
    Matrix& operator--();
    //后缀
    Matrix operator--(int);

    //重载赋值运算符
    Matrix& operator=(const Matrix & mat);
    
    //重载<<运算符
    template<class TT>
    friend std::ostream & operator<<(std::ostream &os, Matrix<TT>& mat);

    //矩阵元素访问运算符
    T& operator()(int r, int c, int ch);

    //相等运算符
    bool operator==(Matrix& mat);

    //返回矩阵里面元素个数
    int size(){
        return row * col;
    }
    //判断矩阵是否为空
    bool empty(){
        return size() == 0;
    }
    
    //返回矩阵通道个数
    int channel_num(){
        return channel;
    }
    
    //ROI相关操作
    //返回子矩阵
    Matrix ROI()const;
    //调整子矩阵位置
    void setROI(int row, int col, int ROIRow, int ROICol);
    void setROIPosition(int row, int col);
    void setROISize(int r, int c);

    //向矩阵尾部添加若干行数据
    Matrix& append(T *nums, int r, int c, int ch);
    //向矩阵尾部删除若干行数据
    Matrix& subtract(int r);
    //矩阵合并(纵向)
    Matrix merge_vertical(const Matrix& mat) const;
    //矩阵合并（横向）
    Matrix merge_horizontal(const Matrix& mat) const;

};

//将静态类成员初始化
template<class T>
int Matrix<T>::num_matrices = 0;

//返回当前矩阵对象的个数
template<typename T>
int Matrix<T>::Matrix_num(){
    return num_matrices;
}

//默认构造器
template<typename T>
Matrix<T>::Matrix(int row, int col, int channel, T* nums, int ROIBeginRow, int ROIBeginCol, int ROIRow, int ROICol, int* ref_count, int span ){
    /*输出检查传入参数
    printf("channel = %d, row = %d, col = %d\nThe element array is \n", channel, row, col);
    for(int i = 0; i < row * col * channel; i++){
        std::cout << nums[i] << " ";
    }
    std::cout << std::endl;*/
    // std::cout << "Default constructor is invoked.\n";
    //参数检查合法性
    if(row < 0 || col < 0 || ROIBeginCol < 0 || ROIBeginRow < 0 || ROIRow < 0 || ROICol < 0){
        std::cerr << "In default constructor" << std::endl;
        std::cerr << "The input of rows and columns should not less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(channel < 1 || channel > 3){
        std::cerr << "In default constructor" << std::endl;
        std::cerr << "The input of channel should between 1 and 3" << std::endl;
        exit(EXIT_FAILURE);
    }
    //如果该矩阵的元素数组没有被引用过，则初始化为1，否则将引用指针指向传入指针
    if(ref_count == nullptr){
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
    }else{
        this -> ref_count = ref_count;
    }

    //对ROI的边界的合法性进行检查
    if(ROIBeginRow + ROIRow > row || ROIBeginCol + ROICol > col){
        std::cerr << "In default constructor" << std::endl;
        std::cerr << "ROIBeginRow = " << ROIBeginRow << " and ROIRow = " << ROIRow << " and row = " << row << std::endl;
        std::cerr << "ROIBeginCol = " << ROIBeginCol << " and ROICol = " << ROICol << " and col = " << col << std::endl;
        std::cerr << "The region of interest should not exceed the region of the matrix." << std::endl;
        exit(EXIT_FAILURE);
    }

    //成员赋值
    this -> row = row;
    this -> col = col;
    this -> channel = channel;
    this -> ROIBeginRow = ROIBeginRow;
    this -> ROIBeginCol = ROIBeginCol;
    this -> ROIRow = ROIRow;
    this -> ROICol = ROICol;
    if(span == 0){
        this -> span = channel * col;
    }else{
        this -> span = span;
    }
    
    //根据通道数创建元素数组
    if(nums == nullptr){
        this -> nums = new T[row * col * channel];
    }else{
        int ele_num = row * col * channel;
        this -> nums = new T[row * col * channel];
        for(int i = 0; i < ele_num; i++){
            this -> nums[i] = nums[i];
        }
    }
    num_matrices ++;
    // std::cout << num_matrices << " member exist \n";

}

template<typename T>
Matrix<T>::Matrix(int row, int col, int channel, std::ifstream & fin, int size){
    
    //参数正确性检查
    if(row < 0 || col < 0){
        std::cerr << "In default constructor" << std::endl;
        std::cerr << "The input of rows and columns should not less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(channel < 1 || channel > 3){
        std::cerr << "In default constructor" << std::endl;
        std::cerr << "The input of channel should between 1 and 3" << std::endl;
        exit(EXIT_FAILURE);
    }
    this -> ref_count = new int[1];
    this -> ref_count[0] = 1;

    //对于传入的文件当中包含的数组大小进行检查
    if(row * col * channel != size){
        std::cerr << "In default constructor" << std::endl;
        std::cerr << "The row = " << row << " and the column = " << col << " and the channel = " << channel << std::endl;
        std::cerr << "The input array's size should equal to the size of matrix's element array." << std::endl;
        exit(EXIT_FAILURE);
    }

    

    //成员赋值
    this -> row = row;
    this -> col = col;
    this -> channel = channel;
    this -> ROIBeginRow = 0;
    this -> ROIBeginCol = 0;
    this -> ROIRow = 0;
    this -> ROICol = 0;
    this -> span = col * channel;

    this -> nums = new T[size];
    int pos = 0;
    
    while(!fin.eof()){
        std::string line;
        getline(fin, line);
        std::stringstream ss(line);
        T f;
        while(ss >> f){
            this -> nums[pos++] = f;
        }    
    }
    // printf("out\n");
    num_matrices ++;

}
    

//复制构造函数
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& mat){
    // std::cout << "Copy constructor is invoked.\n";
    
    this -> row = mat.row;
    this -> col = mat.col;
    this -> channel = mat.channel;
    this -> ROIBeginRow = mat.ROIBeginRow;
    this -> ROIBeginCol = mat.ROIBeginCol;
    this -> ROIRow = mat.ROIRow;
    this -> ROICol = mat.ROICol;
    this -> ref_count = mat.ref_count;
    this -> span = mat.span;

    this -> ref_count[0] ++;
    this -> nums = mat.nums;
    
    num_matrices ++;
    // std::cout << num_matrices << " members exist\n";
    // std::cout << ref_count[0] << " matrices share the element array.\n";

}

//析构函数
template<typename T>
Matrix<T>::~Matrix(){
    // printf("Destructor is invoked.\n");
    num_matrices--;
    // std::cout << num_matrices << " member exist\n";

    if(this -> ref_count[0] == 1){
        // printf("In ref_count\n");
        delete[] this -> ref_count;
        this -> ref_count = nullptr;
        // printf("After deleting ref_count pointer.\n");
        delete[] this -> nums;
        this -> nums = nullptr;
        // std::cout << "The element array has been free.\n";
    }
        
    else{
        this -> ref_count[0]--;
        // std::cout << this -> ref_count[0] << " matrices share the element array.\n";
    }
        
}

//重载赋值运算符
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> & mat){
    // printf("Assignment operator is invoked.\n");
    //参数检查
    
    //防止自反赋值
    if(this == &mat){
        return *this;
    }
    //如果不是自反赋值，则释放原有对象成员指针指向的内存并将非指针变量一一赋值
    if(this -> ref_count[0] == 1){
        delete[] nums;
        // std::cout << "The original element array has been free.\n";
    }else{
        this -> ref_count[0] --;
    }
    
    this -> row = mat.row;
    this -> col = mat.col;
    this -> channel = mat.channel;
    this -> ROIBeginRow = mat.ROIBeginRow;
    this -> ROIBeginCol = mat.ROIBeginCol;
    this -> ROIRow = mat.ROIRow;
    this -> ROICol = mat.ROICol;
    this -> span = mat.span;

    this -> ref_count = mat.ref_count;

    this -> ref_count[0] ++;
    // std::cout << this -> ref_count[0] << " matrices share the element array.\n";

    this -> nums = mat.nums;

    return *this;

}

//加法运算符重载
template <typename T>
Matrix<T> Matrix<T>::operator+(T num) const{
    Matrix<T> mat;
    mat.row = row;
    mat.col = col;
    mat.channel = channel;
    mat.ROIBeginRow = ROIBeginRow;
    mat.ROIBeginCol = ROIBeginCol;
    mat.ROIRow = ROIRow;
    mat.ROICol = ROICol;
    mat.span = span;
    mat.nums = new T[row * col * channel];

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                mat.nums[i * mat.span + j * mat.channel + ch - 1] = this -> nums[i * this -> span + j * this -> channel + ch -1] + num;
                // mat(i, j, ch) = (*this)(i, j, ch) + num;
            }
        }
    }
    return mat;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &m)const{
    
    //参数合法性检查
    if(row != m.row || col != m.col || channel != m.channel){
        std::cerr << "In '+' operator overloading..." << std::endl;
        std::cerr << "The size of the two matrix should be the same." << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix<T> mat;
    mat.row = row;
    mat.col = col;
    mat.ROIBeginRow = ROIBeginRow;
    mat.ROIBeginCol = ROIBeginCol;
    mat.ROIRow = ROIRow;
    mat.ROICol = ROICol;
    mat.channel = channel;
    mat.span = span;

    int ele_num = row * col * channel;
    mat.nums = new T[ele_num];

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                mat.nums[i * mat.span + j * mat.channel + ch - 1] = 
                this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                + m.nums[i * m.span + j * m.channel + ch - 1];
            }
        }
    }
    return mat;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T> &mat){
    //参数检查
    if(row != mat.row || col != mat.col || channel != mat.channel){
        std::cerr << "In assignment operator overloading..." << std::endl;
        std::cerr << "The number of row and col and channel of the two matrix should be the same." << std::endl;
        exit(EXIT_FAILURE);
    }

    int ele_num = row * col * channel;//总的元素个数
    if(this -> ref_count[0] == 1){//如果引用次数为1，则直接改变原数组的元素值

        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    + mat.nums[i * mat.span + j * mat.channel + ch - 1];
                }
            }
        }
    }else{//如果引用数不为1，则需要重新开辟一块新的空间用于存储新的元素值
        this -> ref_count[0] --;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
        T* n = new T[ele_num];//开辟一块新的空间存储元素数组
        int pos = 0;
       
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    n[pos++] = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                + mat.nums[i * mat.span + j * mat.channel + ch - 1];
                }
            }
        }
        this -> nums = n;
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(T num){
    int ele_num = row * col * channel;
    if(this -> ref_count[0] == 1){
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    + num;
                }
            }
        }
    }else{
        this -> ref_count[0]--;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
        T* n = new T[ele_num];
        int pos = 0;
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    n[pos++] = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    + num;
                }
            }
        }
        this -> nums = n;
    }
    
    return *this;
}

//减法运算符重载
template <typename T>
Matrix<T> Matrix<T>::operator-(T num) const{
    Matrix<T> mat;
    mat.row = row;
    mat.col = col;
    mat.channel = channel;
    mat.ROIBeginRow = ROIBeginRow;
    mat.ROIBeginCol = ROIBeginCol;
    mat.ROIRow = ROIRow;
    mat.ROICol = ROICol;
    mat.span = span;
    mat.nums = new T[row * col * channel];

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                mat.nums[i * mat.span + j * mat.channel + ch - 1] = this -> nums[i * this -> span + j * this -> channel + ch -1] - num;
                
            }
        }
    }
    return mat;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &m)const{
    
    //参数合法性检查
    if(row != m.row || col != m.col || channel != m.channel){
        std::cerr << "In '+' operator overloading..." << std::endl;
        std::cerr << "The size of the two matrix should be the same." << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix<T> mat;
    mat.row = row;
    mat.col = col;
    mat.ROIBeginRow = ROIBeginRow;
    mat.ROIBeginCol = ROIBeginCol;
    mat.ROIRow = ROIRow;
    mat.ROICol = ROICol;
    mat.channel = channel;
    mat.span = span;

    int ele_num = row * col * channel;
    mat.nums = new T[ele_num];

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                mat.nums[i * mat.span + j * mat.channel + ch - 1] = 
                this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                - m.nums[i * m.span + j * m.channel + ch - 1];
            }
        }
    }
    return mat;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T> &mat){
     //参数检查
    if(row != mat.row || col != mat.col || channel != mat.channel){
        std::cerr << "In assignment operator overloading..." << std::endl;
        std::cerr << "The number of row and col and channel of the two matrix should be the same." << std::endl;
        exit(EXIT_FAILURE);
    }

    int ele_num = row * col * channel;//总的元素个数
    if(this -> ref_count[0] == 1){//如果引用次数为1，则直接改变原数组的元素值

        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    - mat.nums[i * mat.span + j * mat.channel + ch - 1];
                }
            }
        }
    }else{//如果引用数不为1，则需要重新开辟一块新的空间用于存储新的元素值
        this -> ref_count[0] --;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
        T* n = new T[ele_num];//开辟一块新的空间存储元素数组
        int pos = 0;
       
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    n[pos++] = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                - mat.nums[i * mat.span + j * mat.channel + ch - 1];
                }
            }
        }
        this -> nums = n;
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(T num){
    int ele_num = row * col * channel;
    if(this -> ref_count[0] == 1){
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    - num;
                }
            }
        }
    }else{
        this -> ref_count[0]--;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
        T* n = new T[ele_num];
        int pos = 0;
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    n[pos++] = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    - num;
                }
            }
        }
        this -> nums = n;
    }
    
    return *this;
}

//除法运算符重载
template <typename T>
Matrix<T> Matrix<T>::operator/(T num) const{
    Matrix<T> mat;
    mat.row = row;
    mat.col = col;
    mat.channel = channel;
    mat.ROIBeginRow = ROIBeginRow;
    mat.ROIBeginCol = ROIBeginCol;
    mat.ROIRow = ROIRow;
    mat.ROICol = ROICol;
    mat.span = span;
    mat.nums = new T[row * col * channel];

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                mat.nums[i * mat.span + j * mat.channel + ch - 1] = this -> nums[i * this -> span + j * this -> channel + ch -1] / num;
                
            }
        }
    }
    return mat;
}

template<typename T>
Matrix<T>& Matrix<T>::operator/=(T num){
    int ele_num = row * col * channel;
    if(this -> ref_count[0] == 1){
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    / num;
                }
            }
        }
    }else{
        this -> ref_count[0]--;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
        T* n = new T[ele_num];
        int pos = 0;
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    n[pos++] = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    / num;
                }
            }
        }
        this -> nums = n;
    }
    
    return *this;
}

//乘法运算符重载
template <typename T>
Matrix<T> Matrix<T>::operator*(T num)const{
    Matrix mat;
    mat.row = row;
    mat.col = col;
    mat.channel = channel;
    mat.ROIBeginRow = ROIBeginRow;
    mat.ROIBeginCol = ROIBeginCol;
    mat.ROIRow = ROIRow;
    mat.ROICol = ROICol;
    mat.span = span;
    mat.nums = new T[row * col * channel];

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                mat.nums[i * mat.span + j * mat.channel + ch - 1] = this -> nums[i * this -> span + j * this -> channel + ch -1] * num;
            }
        }
    }
    num_matrices++;
    return mat;
}
template<typename T>
Matrix<T>& Matrix<T>::operator*=(T num){
    int ele_num = row * col * channel;
    if(this -> ref_count[0] == 1){
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    * num;
                }
            }
        }
    }else{
        this -> ref_count[0]--;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
        T* n = new T[ele_num];
        int pos = 0;
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    n[pos++] = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    * num;
                }
            }
        }
        this -> nums = n;
    }
    
    return *this;
}

//矩阵乘法
template<typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T> &mat){
   
    //判断参数合法性
    if(mat.row != this -> col || mat.channel != channel){
        std::cerr << "In operator * friend functoin..." << std::endl;
        std::cerr << "The two matrices' sizes and channel number should be the same." << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix<T> res;
    num_matrices++;
    res.row = row;
    res.col = mat.col;
    res.channel = channel;
    res.ROIBeginRow = 0;
    res.ROIBeginCol = 0;
    res.ROIRow = 0;
    res.ROICol = 0;
    res.span = res.channel * res.col;
    int ele_num = res.row * res.col * res.channel;
    res.nums = new T[ele_num];
    /*
    for(int ch = 1; ch <= channel; ch ++){//不同通道数的切换
         for(int i = 0; i < row; i ++){
            for(int j = 0; j < mat.col; j ++){
                for(int k = 0; k < mat.row; k ++){ 
                    res(i, j, ch) += (*this)(i, k, ch) * mat(k, j, ch);
                }                    
            }
        }
    }*/
    /*
    //不同通道数的切换
    for(int i = 0; i < row; i ++){
        for(int j = 0; j < mat.col; j ++){
            for(int k = 0; k < mat.row; k ++){ 
                for(int ch = 1; ch <= channel; ch ++){
                    res(i, j, ch) += (*this)(i, k, ch) * mat(k, j, ch);
                }                    
            }
        }
    }
    */

    /**/
    //不同通道数的切换
    for(int k = 0; k < mat.row; k ++){ 
        for(int i = 0; i < row; i ++){
            for(int j = 0; j < mat.col; j ++){
                for(int ch = 1; ch <= channel; ch ++){
                    res(i, j, ch) += (*this)(i, k, ch) * mat(k, j, ch);
                }                    
            }
        }
    }
    
    return res;
}


//矩阵转置运算符
template<typename T>
Matrix<T> Matrix<T>::operator!() const {
    Matrix<T> mat;
    mat.row = this -> col;
    mat.col = this -> row;
    mat.ROIBeginRow = this -> ROIBeginCol;
    mat.ROIBeginCol = this -> ROIBeginRow;
    mat.ROIRow = this -> ROICol;
    mat.ROICol = this -> ROIRow;
    mat.channel = this -> channel;
    int ele_num = mat.row * mat.col * mat.channel;
    mat.nums = new T[ele_num];
    mat.span = mat.col * mat.channel;
    

    for(int i = 0; i < mat.row; i++){
        for(int j = 0; j < mat.col; j++){
            for(int k = 1; k <= mat.channel; k++){
                mat(i, j, k) = this -> nums[j * this -> span + i * this -> channel + k - 1];
            }
        }
    }
    return mat;
}

//前缀自增运算符
template<typename T>
Matrix<T>& Matrix<T>::operator++(){
    int ele_num = row * col * channel;//总的元素个数
    if(this -> ref_count[0] == 1){//如果引用次数为1，则直接改变原数组的元素值

        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    + 1;
                }
            }
        }
    }else{//如果引用数不为1，则需要重新开辟一块新的空间用于存储新的元素值
        this -> ref_count[0] --;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
        T* n = new T[ele_num];//开辟一块新的空间存储元素数组
        int pos = 0;
       
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    n[pos++] = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                + 1;
                }
            }
        }
        this -> nums = n;
    }
    return *this;
}
//后缀自增运算符
template<typename T>
Matrix<T> Matrix<T>::operator++(int){
    Matrix<T> old = *this; // 保持初始值
    operator++();  // 前缀自增
    return old; //返回初始值
}

//前缀自减运算符
template<typename T>
Matrix<T>& Matrix<T>::operator--(){
    int ele_num = row * col * channel;//总的元素个数
    if(this -> ref_count[0] == 1){//如果引用次数为1，则直接改变原数组的元素值

        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                    - 1;
                }
            }
        }
    }else{//如果引用数不为1，则需要重新开辟一块新的空间用于存储新的元素值
        this -> ref_count[0] --;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
        T* n = new T[ele_num];//开辟一块新的空间存储元素数组
        int pos = 0;
       
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                for(int ch = 1; ch <= channel; ch++){
                    n[pos++] = this -> nums[i * this -> span + j * this -> channel + ch - 1] 
                - 1;
                }
            }
        }
        this -> nums = n;
    }
    return *this;
}
//后缀自减运算符
template<typename T>
Matrix<T> Matrix<T>::operator--(int){
    Matrix<T> old = *this; // 保持初始值
    operator--();  // 前缀自增
    return old; //返回初始值
}
//矩阵元素访问运算符
template<typename T>
T & Matrix<T>::operator()(int r, int c, int ch){
    //参数合法性检查
    if(r < 0 || c < 0){
        std::cerr << "The input of row and column should not be less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(ch < 1 || ch > channel){
        std::cerr << "The input of channel should be between 1 and 3" << std::endl;
        exit(EXIT_FAILURE);
    }

    return nums[r * span + c * channel + ch - 1];
}


//相等运算符重载
template<typename T>
bool Matrix<T>::operator==(Matrix<T>& mat){

    if(row != mat.row || col != mat.col || ROIBeginRow != mat.ROIBeginRow || ROIBeginCol != mat.ROIBeginCol
    || ROIRow != mat.ROIRow || ROICol != mat.ROIRow || channel != mat.channel){
        return false;
    }
    

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                
                if((i, j, ch) != mat(i, j, ch)) {
                    return false;
                }
            }
        }
    }

    return true;

}

//重载<<运算符
template<class T>
std::ostream & operator<<(std::ostream &os, Matrix<T>& mat){
    int ch = mat.channel;
    int r = mat.row;
    int c = mat.col;
    os << "[";
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            os << "{";
            for(int k = 1; k <= ch; k++){
                os << mat(i, j, k);
                if(k != ch)  os << " ";
            }
            os << "}";
            if(j != c - 1) os << ", ";
        }
        if(i != r - 1) os << "\n";
    }
    os << "]";
    return os; 
}

//ROI相关操作
//返回子矩阵
template<typename T>
Matrix<T> Matrix<T>::ROI() const{
    //对ROI的边界的合法性进行检查
    if(ROIBeginRow + ROIRow > row || ROIBeginCol + ROICol > col){
        std::cerr << "The region of interest should not exceed the region of the matrix." << std::endl;
        exit(EXIT_FAILURE);
    }
    Matrix mat;
    num_matrices++;
    mat.row = ROIRow;
    mat.col = ROICol;
    mat.channel = channel;
    /*
    mat.nums = this -> nums + ROIBeginRow * col * channel + ROIBeginCol * channel;
    mat.ref_count = this -> ref_count;
    mat.ref_count[0] ++;*/

    int pos =0;
    int ele_num = mat.row * mat.col * mat.channel;
    mat.nums = new T[ele_num];
    for(int i = ROIBeginRow; i < ROIBeginRow + ROIRow; i++){
        for(int j = ROIBeginCol; j < ROIBeginCol + ROICol; j ++){
            for (int k = 1; k <= channel; k++) {
                mat.nums[pos++] = nums[i * ROICol * channel + j * channel + k - 1];
            }     
        }
    }
    // std::cout << mat.ref_count[0] << " matrix share the array space." << std::endl;
    mat.span = this -> col * this -> channel;
    std::cout << "Finish creating sub matrix." << std::endl;
    return mat;

}

template<typename T>
void Matrix<T>::setROI(int row, int col, int ROIRow, int ROICol){
    //参数正确性检查
    if(row < 0 || col < 0 || ROIRow < 0 || ROICol < 0){
        std::cerr << "In function setROI..." << std::endl;
        std::cerr << "The input of row and column should not be less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }

    this -> ROIBeginCol = col;
    this -> ROIBeginRow = row;
    this -> ROIRow = ROIRow;
    this -> ROICol = ROICol;
}

template<typename T>
void Matrix<T>::setROIPosition(int row, int col){
    //参数正确性检查
    if(row < 0 || col < 0){
        std::cerr << "In function setROIPosition..." << std::endl;
        std::cerr << "The input of row and column should not be less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(row > this -> row || col > this -> col){
        std::cerr << "In function setROIPosition..." << std::endl;
        std::cerr << "The input of row and column should not be larger than the row number or colomn number of the source matrix." << std::endl;
        exit(EXIT_FAILURE);
    }
    //赋值
    this -> ROIBeginRow = row;
    this -> ROIBeginCol = col;
}
template<typename T>
void Matrix<T>::setROISize(int r, int c){
    //参数正确性检查
    if(r < 0 || c < 0){
        std::cerr << "In function setROISize..." << std::endl;
        std::cerr << "The input of row and column should not be less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    this -> ROIRow = r;
    this -> ROICol = c;
};

//向矩阵尾部添加若干行数据
template<typename T>
Matrix<T>& Matrix<T>::append(T *nums, int r, int c, int ch){
    
    //参数合法性检查
    if(nums == NULL){
        std::cerr << "In function append..." << std::endl;
        std::cerr << "The input array should be valid." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(r <= 0 || c <= 0){
        std::cerr << "In function append..." << std::endl;
        std::cerr << "The input of row and column should not be less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(c != col || ch != channel){
        std::cerr << "In function append..." << std::endl;
        std::cerr << "The channel and column should equal to the source matrix's channel and column." << std::endl;
        exit(EXIT_FAILURE);
    }


    int src_ele_num = row * col * channel;//原矩阵元素个数
    int tar_ele_num = r * c * ch;//新传入元素数组的大小数组的大小
    T * arr = new T[src_ele_num + tar_ele_num];//创建新元素数组，大小为两个数组大小之和
    int pos = 0;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                arr[pos++] = (*this)(i, j, ch);
            }
        }
    }
    for(int j = 0; j < tar_ele_num; j++){
        arr[pos++] = nums[j];
        //在新数组后方赋值为新传入的数组
    }
    if(this -> ref_count[0] == 1){
        delete[] this -> nums;//防止内存泄漏，释放源矩阵原有的元素数组所占空间
    }else{//将原有引用值加一，并将新的引用值设置为1
        this -> ref_count[0]--;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
    }
    
    this -> nums = arr;//将矩阵对象的元素数组指向新开辟的内存区域
    this -> row = this -> row + r;
    return *this;

}

//向矩阵尾部删除若干行数据
template<typename T>
Matrix<T>& Matrix<T>::subtract(int r){
    if(r < 0){
        std::cerr << "In function subtract..." << std::endl;
        std::cerr << "The input of delete row should not be less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(r > this -> row){
        std::cerr << "In function subtract..." << std::endl;
        std::cerr << "The input of delete row should not be larger than the source matrix's row number." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int ele_num = (row - r) * col * channel;//新数组大小
    T* arr = new T[ele_num];
    int pos = 0;
    for(int i = 0; i < row - r; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                arr[pos++] = (*this)(i, j, ch);
            }
        }
    }
    if(this -> ref_count[0] == 1){
        delete[] this -> nums;//防止内存泄漏，释放源矩阵原有的元素数组所占空间
    }else{//将原有引用值加一，并将新的引用值设置为1
        this -> ref_count[0]--;
        this -> ref_count = new int[1];
        this -> ref_count[0] = 1;
    }
    this -> nums = arr;//赋值
    this -> row -= r;
    return *this;
}

//矩阵合并(纵向)
template <typename T>
Matrix<T> Matrix<T>::merge_vertical(const Matrix<T>& mat) const {
    //检查参数合法性
    
    //传入矩阵成员数组为空的不合法情况
    if(mat.nums == nullptr){
        std::cerr << "In function merge_vertical..." << std::endl;
        std::cerr << "The elements array of source matrix should be valid" << std::endl;
        exit(EXIT_FAILURE);
    }
    //两个矩阵规模或者大小不一致的情况
    if(this -> col != mat.col || this -> channel != mat.channel){
        std::cerr << "In function merge_vertical..." << std::endl;
        std::cerr << "The column and channel of the two matrix should be the same." << std::endl;
        exit(EXIT_FAILURE);
    }

    int row1 = row;
    int row2 = mat.row;

    Matrix<T> res;
    num_matrices++;
    res.row = row1 + row2;
    res.col = col;
    res.channel = channel;
    res.nums = new T[res.row * col * channel];
    res.span = res.col * res.channel;

    int pos = 0;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int ch = 1; ch <= channel; ch++){
                res(i, j, ch) = this -> nums[i * this -> span + j * this -> channel + ch - 1];
                // res(i, j, ch) = this -> nums[i * this -> span + j * this -> channel + ch - 1];
            }
        }
    }
    

    for(int i = 0; i < mat.row; i++){
        for(int j = 0; j < mat.col; j++){
            for(int ch = 1; ch <= mat.channel; ch++){
                res(i + row1, j, ch) = mat.nums[i * mat.span + j * mat.channel + ch - 1];
            }
        }
    }

    return res;

}
//矩阵的合并(横向)
template<typename T>
Matrix<T> Matrix<T>::merge_horizontal(const Matrix<T> & mat)const {
    //检查参数合法性
    
    //传入矩阵成员数组为空的不合法情况
    if(mat.nums == nullptr){
        std::cerr << "In function merge_vertical..." << std::endl;
        std::cerr << "The elements array of source matrix should be valid" << std::endl;
        exit(EXIT_FAILURE);
    }
    //两个矩阵规模或者大小不一致的情况
    if(row != mat.row || channel != mat.channel){
        std::cerr << "In function merge_vertical..." << std::endl;
        std::cerr << "The row and channel of the two matrix should be the same." << std::endl;
        exit(EXIT_FAILURE);
    }
    Matrix<T> res;
    num_matrices++;
    res.row = row;
    res.col = col + mat.col;
    res.channel = channel;
    res.span = res.col * res.channel;
    res.nums = new T[row * res.col * channel];

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col ; j++){
            for(int ch = 1; ch <= channel; ch++){
                // res(i, j, ch) = (*this)(i, j, ch);
                res(i, j, ch) = this -> nums[i * this -> span + j * this -> channel + ch - 1];
            }
        }
        
        for(int j = 0; j < mat.col ; j++){
            for(int ch = 1; ch <= channel; ch++){
                // res(i, col + j, ch) = mat(i, j, ch);
                res(i, col + j, ch) = mat.nums[i * mat.span + j * mat.channel + ch - 1];
            }
        }
    }

    return res;

}


//向量点乘运算
template<typename T>
T Matrix<T>::dotmul(int r1, T* nums1, int r2, T* nums2){

    //参数检查合法性
    if(r1 < 0 || r2 < 0) {
        std::cerr << "Break in dotmul." << std::endl;
        std::cerr << "The input of r1 and r2 should not less than 0." << std::endl;
        exit(EXIT_FAILURE);
    }
    if(r1 != r2) {
        std::cerr << "Break in dotmul." << std::endl;
        std::cerr << "The input of r1 and r2 should be equivalent." << std::endl;
        exit(EXIT_FAILURE);
    }
    //检测空指针
    if(nums1 == nullptr || nums2 == nullptr) {
        std::cerr << "Break in dotmul." << std::endl;
        std::cerr << "The two vectors should be valid." << std::endl;
    }
    //矩阵点乘运算
    T result = 0;
    for(int i = 0; i < r1; i++) {
        result += nums1[i] * nums2[i];
    }
    return result;
}

#endif