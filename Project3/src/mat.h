#ifndef _MATMUL_F
#include <stdio.h>
#include<math.h>

#include <stdbool.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include<time.h>
#define _MATMUL_F

struct Matrix{
    int row; 
    int col;
    float * nums;
};

struct Matrix *init_file(struct Matrix *src, int r, int c, FILE *fp); //初始化矩阵

struct Matrix *init(struct Matrix *src, int r, int c);

bool delMatrix(struct Matrix *mat); //删除矩阵

bool cpyMatrix(struct Matrix *mat1, struct Matrix *mat2); //复制矩阵

bool mulMatrix_tradition(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2); //矩阵乘法

bool mulMatrix_ikj(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2);

bool mulMatrix_kij(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2);

bool mulMatrix_block(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2);

bool mulMatrix_strassen(struct Matrix *tar, struct Matrix *mat1, struct Matrix *mat2, int size);

bool mat_add(struct Matrix * res, struct Matrix *mat1, struct Matrix* mat2, int size);

bool mat_sub(struct Matrix * res, struct Matrix * mat1, struct Matrix * mat2, int size);

int matSize(char str[]); //通过文件名获取矩阵规模

void random_mat(struct Matrix* mat);


#endif