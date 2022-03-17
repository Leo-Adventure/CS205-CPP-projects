#include "mat.h"


struct Matrix * init_file(struct Matrix* mat, int r, int c, FILE* fp){
    if(r <= 0 || c <= 0){
        printf("The line or colomn input is improper!\n");
        return NULL;
    }

    //为矩阵结构各结构体变量赋值
    mat -> row = r;
    mat -> col = c;
    mat -> nums = (float *)malloc(r * c * sizeof(float));//动态分配二维数组的内存
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            fscanf(fp, "%f", &(mat -> nums[i * c + j]));
        }
    };

    fclose(fp);//关闭文件
    return mat;

}

struct Matrix* init(struct Matrix* mat, int r, int c){
    if(r <= 0 || c <= 0){
        printf("The line or colomn input is improper!\n");
        return NULL;
    }

    //为矩阵结构各结构体变量赋值
    mat -> row = r;
    mat -> col = c;
    mat -> nums = (float *)malloc(r * c * sizeof(float));//动态分配二维数组的内存
    return mat;

}


bool delMatrix(struct Matrix* mat){
    if(mat == NULL){
        return false;
    }
    
    free(mat -> nums);
    //即使内存已经释放，但指针仍然指向原有内存，应该置零
    mat -> nums = NULL;
    free(mat);
    mat = NULL;
    return true;

}

int matSize(char str[]){
    char *delim = "-.";
    char *p;
    strtok(str, delim);
    int cnt = 0;
    
    while ((p = strtok(NULL, delim))){
        if(cnt == 1){
            break;
        }
        cnt++;
    }
    return atoi(p);   
}

bool cpyMatrix(struct Matrix *target, struct Matrix *src){
    if(src == NULL){
        return false;
    }
    
    int row = src -> row;
    int col = src -> col;

    if(target == NULL && target -> row != row || target -> col != col){
        delMatrix(target);
        struct Matrix *tar;
        tar = (struct Matrix*)malloc(sizeof(struct Matrix));
        tar -> row = row;
        tar -> col = col;
        tar -> nums = (float*)malloc(row * col * sizeof(float));
        target = tar;
    }
    

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            target -> nums[i * col + j] = src -> nums[i * col + j];
        }
    }
    return true;
}

bool mulMatrix_tradition(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2){
    if(mat1 == NULL || mat2 == NULL || mat1 -> row <= 0 || mat2 -> row <= 0 || mat1 -> col <= 0 || mat2 -> col <= 0){
        return false;
    }
    if(mat1 -> row != mat1 -> col || mat2 -> row != mat2 -> col || mat1 -> row != mat2 -> row){
        return false;
    }

    int row = mat1 -> row;
    int col = mat1 -> col;
    
    if(tar == NULL || tar -> row != row || tar -> col != col){
        delMatrix(tar);
        struct Matrix *tar_ex;
        tar_ex = (struct Matrix*)malloc(sizeof(struct Matrix));
        tar_ex -> row = row;
        tar_ex -> col = col;
        tar_ex -> nums = (float*)malloc(row * col * sizeof(float));
        tar = tar_ex;
    }

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int k = 0; k < row; k++){
                tar -> nums[i * col + j] += mat1 -> nums[i * col + k] * mat2 -> nums[k * col + j];
            }
        }
    }

    return true;
}

bool mulMatrix_block(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2){
    if(mat1 == NULL || mat2 == NULL || mat1 -> row <= 0 || mat2 -> row <= 0 || mat1 -> col <= 0 || mat2 -> col <= 0){
        return false;
    }
    if(mat1 -> row != mat1 -> col || mat2 -> row != mat2 -> col || mat1 -> row != mat2 -> row){
        return false;
    }

    int row = mat1 -> row;
    int col = mat1 -> col;
    
    if(tar == NULL || tar -> row != row || tar -> col != col){
        delMatrix(tar);
        struct Matrix *tar_ex;
        tar_ex = (struct Matrix*)malloc(sizeof(struct Matrix));
        tar_ex -> row = row;
        tar_ex -> col = col;
        tar_ex -> nums = (float*)malloc(row * col * sizeof(float));
        tar = tar_ex;
    }

    int block_size = 4;
    
    for (int block_i = 0; block_i < row; block_i += block_size) {
        for (int block_j = 0; block_j < row; block_j += block_size) {
            for (int block_k = 0; block_k < row; block_k += block_size) {
                for (int i = block_i; i < (block_i + block_size >= row)?row: (block_i+ block_size); i++) {
                    for (int j = block_j; j < (block_j + block_size >=row)?row:(block_j + block_size); j++) {
                        for (int k = block_k; k < (block_k + block_size >= row)?row: (block_k + block_size); k++) {
                            printf("%f, %f\n", mat1 -> nums[i * col + k], mat2 -> nums[k * col + j]);
                            tar -> nums[i * col + j] += mat1 -> nums[i * col + k] * mat2 -> nums[k * col + j];
                            printf("success\n");
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool mulMatrix_ikj(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2){
    if(mat1 == NULL || mat2 == NULL || mat1 -> row <= 0 || mat2 -> row <= 0 || mat1 -> col <= 0 || mat2 -> col <= 0){
        return false;
    }
    if(mat1 -> row != mat1 -> col || mat2 -> row != mat2 -> col || mat1 -> row != mat2 -> row){
        return false;
    }

    int row = mat1 -> row;
    int col = mat1 -> col;
    
    if(tar == NULL || tar -> row != row || tar -> col != col){
        delMatrix(tar);
        struct Matrix *tar_ex;
        tar_ex = (struct Matrix*)malloc(sizeof(struct Matrix));
        tar_ex -> row = row;
        tar_ex -> col = col;
        tar_ex -> nums = (float*)malloc(row * col * sizeof(float));
        tar = tar_ex;
    }

    for(int i = 0; i < row; i++){
        for(int k = 0; k < row; k++){
            float tmp = mat1 -> nums[i * col + k];
            for(int j = 0; j < col; j++){
                tar -> nums[i * col + j] += tmp * mat2 -> nums[k * col + j];
            }
        }
    }

    return true;
}

bool mulMatrix_kij(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2){
    if(mat1 == NULL || mat2 == NULL || mat1 -> row <= 0 || mat2 -> row <= 0 || mat1 -> col <= 0 || mat2 -> col <= 0){
        return false;
    }
    if(mat1 -> row != mat1 -> col || mat2 -> row != mat2 -> col || mat1 -> row != mat2 -> row){
        return false;
    }

    int row = mat1 -> row;
    int col = mat1 -> col;
    
    if(tar == NULL || tar -> row != row || tar -> col != col){
        delMatrix(tar);
        struct Matrix *tar_ex;
        tar_ex = (struct Matrix*)malloc(sizeof(struct Matrix));
        tar_ex -> row = row;
        tar_ex -> col = col;
        tar_ex -> nums = (float*)malloc(row * col * sizeof(float));
        tar = tar_ex;
    }

    for(int k = 0; k < row; k++){
        for(int i = 0; i < row; i++){
            float tmp = mat1 -> nums[i * col + k];
            for(int j = 0; j < col; j++){
                tar -> nums[i * col + j] += tmp * mat2 -> nums[k * col + j];
            }
        }
    }

    return true;
}

bool mulMatrix_strassen(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2, int size){
    
    int row = mat1 -> row;
    int col = mat1 -> col;
    int newSize = size / 2;
    
    
    if(tar == NULL || tar -> row != row || tar -> col != col){
        delMatrix(tar);
        struct Matrix *tar_ex;
        tar_ex = (struct Matrix*)malloc(sizeof(struct Matrix));
        tar_ex -> row = row;
        tar_ex -> col = col;
        tar_ex -> nums = (float*)malloc(row * col * sizeof(float));
        tar = tar_ex;
    }
    
    if(row <= 32){
        mulMatrix_ikj(tar, mat1, mat2);
        return true;
    }
    
    struct Matrix* A11 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* A12 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* A21 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* A22 = (struct Matrix*) malloc(sizeof(struct Matrix));

    init(A11, newSize, newSize);
    init(A12, newSize, newSize);
    init(A21, newSize, newSize);
    init(A22, newSize, newSize);
    

    struct Matrix* B11 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* B12 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* B21 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* B22 = (struct Matrix*) malloc(sizeof(struct Matrix));

    init(B11, newSize, newSize);
    init(B12, newSize, newSize);
    init(B21, newSize, newSize);
    init(B22, newSize, newSize);
    
    struct Matrix* C11 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* C12 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* C21 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* C22 = (struct Matrix*) malloc(sizeof(struct Matrix));

    init(C11, newSize, newSize);
    init(C12, newSize, newSize);
    init(C21, newSize, newSize);
    init(C22, newSize, newSize);

    struct Matrix* resA = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* resB = (struct Matrix*) malloc(sizeof(struct Matrix));

    init(resA, newSize, newSize);
    init(resB, newSize, newSize);

    
    struct Matrix* P1 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* P2 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* P3 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* P4 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* P5 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* P6 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* P7 = (struct Matrix*) malloc(sizeof(struct Matrix));

    init(P1, newSize, newSize);
    init(P2, newSize, newSize);
    init(P3, newSize, newSize);
    init(P4, newSize, newSize);
    init(P5, newSize, newSize);
    init(P6, newSize, newSize);
    init(P7, newSize, newSize);

    struct Matrix* S1 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S2 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S3 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S4 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S5 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S6 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S7 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S8 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S9 = (struct Matrix*) malloc(sizeof(struct Matrix));
    struct Matrix* S10 = (struct Matrix*) malloc(sizeof(struct Matrix));

    init(S1, newSize, newSize);
    init(S2, newSize, newSize);
    init(S3, newSize, newSize);
    init(S4, newSize, newSize);
    init(S5, newSize, newSize);
    init(S6, newSize, newSize);
    init(S7, newSize, newSize);
    init(S8, newSize, newSize);
    init(S9, newSize, newSize);
    init(S10, newSize, newSize);
    

   
    
    //对矩阵mat1 和 mat2进行分割
    for(int i = 0; i < newSize; i++){
        for(int j = 0; j < newSize; j++){
            A11 -> nums[i * newSize + j] = mat1 -> nums[i * col + j];
            A12 -> nums[i * newSize + j] = mat1 -> nums[i * col + j + row / 2];
            A21 -> nums[i * newSize + j] = mat1 -> nums[(i + newSize) * col + j];
            A22 -> nums[i * newSize + j] = mat1 -> nums[(i + newSize) * col + j + newSize];

            B11 -> nums[i * newSize + j] = mat2 -> nums[i * col + j];
            B12 -> nums[i * newSize + j] = mat2 -> nums[i * col + j + row / 2];
            B21 -> nums[i * newSize + j] = mat2 -> nums[(i + newSize) * col + j];
            B22 -> nums[i * newSize + j] = mat2 -> nums[(i + newSize) * col + j + row / 2];
        }
    }

    //bool mat_add(struct Matrix * res, struct Matrix *mat1, struct Matrix* mat2, int size);
    //bool mat_sub(struct Matrix * res, struct Matrix * mat1, struct Matrix * mat2, int size);
    //bool mulMatrix_strassen(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2, int size);
    /**/
    mat_sub(S1, B12, B22, newSize);

    mat_add(S2, A11, A12, newSize);

    mat_add(S3, A21, A22, newSize);

    mat_sub(S4, B21, B11, newSize);

    mat_add(S5, A11, A22, newSize);

    mat_add(S6, B11, B22, newSize);

    mat_sub(S7, A12, A22, newSize);

    mat_add(S8, B21, B22, newSize);

    mat_sub(S9, A11, A21, newSize);

    mat_add(S10, B11, B12, newSize);


    mulMatrix_strassen(P1, A11, S1, newSize);

    mulMatrix_strassen(P2, S2, B22, newSize);

    mulMatrix_strassen(P3, S3, B11, newSize);

    mulMatrix_strassen(P4, A22, S4, newSize);

    mulMatrix_strassen(P5, S5, S6, newSize);

    mulMatrix_strassen(P6, S7, S8, newSize);

    mulMatrix_strassen(P7, S9, S10, newSize);

    mat_add(resA, P4, P5, newSize);

    mat_sub(resB, resA, P2, newSize);

    mat_add(C11, resB, P6, newSize);

    mat_add(C12, P1, P2, newSize);

    mat_add(C21, P3, P4, newSize);

    mat_add(resA, P5, P1, newSize);

    mat_sub(resB, resA, P3, newSize);

    mat_sub(C22, resB, P7, newSize);
    
    
    for(int i = 0; i < newSize; i++){
        for(int j = 0; j < newSize; j++){
            tar -> nums[i * col + j] = C11 -> nums[i * newSize + j];
            tar -> nums[i * col + j + row / 2] = C12 -> nums[i * newSize + j];
            tar -> nums[(i + row / 2) * col + j] = C21 -> nums[i * newSize + j];
            tar -> nums[(i + row / 2) * col + j + row / 2] = C22 -> nums[i * newSize + j];
        }
    }
   

    // delMatrix(A11);
    // delMatrix(A12);
    // delMatrix(A21);
    // delMatrix(A22);

    // delMatrix(B11);
    // delMatrix(B12);
    // delMatrix(B21);
    // delMatrix(B22);

    // delMatrix(C11);
    // delMatrix(C12);
    // delMatrix(C21);
    // delMatrix(C22);

    // delMatrix(P1);
    // delMatrix(P2);
    // delMatrix(P3);
    // delMatrix(P4);
    // delMatrix(P5);
    // delMatrix(P6);
    // delMatrix(P7);

    // delMatrix(S1);
    // delMatrix(S2);
    // delMatrix(S3);
    // delMatrix(S4);
    // delMatrix(S5);
    // delMatrix(S6);
    // delMatrix(S7);
    // delMatrix(S8);
    // delMatrix(S9);
    // delMatrix(S10);

    // delMatrix(resA);
    // delMatrix(resB);

    return true;

}

bool mat_add(struct Matrix * res, struct Matrix *mat1, struct Matrix* mat2, int size){
    
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            res -> nums[i * (res -> col) + j] = mat1 -> nums[i * (mat1 -> col) + j] + mat2 -> nums[i * (mat2 -> col) + j];
        }
    }
    return true;
}

bool mat_sub(struct Matrix * res, struct Matrix * mat1, struct Matrix * mat2, int size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            res -> nums[i * (res -> col) + j] = mat1 -> nums[i * (mat1 -> col) + j] - mat2 -> nums[i * (mat2 -> col) + j];
        }
    }
    return true;
}


void random_mat(struct Matrix* mat){
    int r = mat -> row;
     
    srand((unsigned)time(NULL));
    for(int i = 0; i < r; i++){
        for(int j = 0; j < r; j++){
            float value = (rand() % (100 * 10 - 1)) / 10.0;
            mat -> nums[i * r + j] = value;   
        }
    }
    
}