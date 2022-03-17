#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include</opt/OpenBLAS/include/cblas.h>
#include "mat.h"
#pragma GCC optimize(2)


int main(int argc, char * argv[]){

    if(argc < 4){
        printf("The file cannot be received normally, please enter files properly\n");
        exit(0);
    }
    
    
    char *ch1 = (char*) malloc(10 * sizeof(char));
    char *ch2 = (char*) malloc(10 * sizeof(char));
    
    strcpy(ch1, argv[1]);
    strcpy(ch2, argv[2]);
    
    int size1 = matSize(ch1);
    int size2 = matSize(ch2);
    

    if(size1 != size2){
        printf("The size of matrix 1 and the size of matrix 2 are not the same\nPlease check again.\n");
        exit(0);
    }

    struct Matrix *mat1 = (struct Matrix*)malloc(sizeof(struct Matrix)); 
    struct Matrix *mat2 = (struct Matrix*)malloc(sizeof(struct Matrix));
    

    FILE *fp1, *fp2;

    fp1 = fopen(argv[1], "r");
    
    if(fp1 == NULL){
        printf("Error in opening the first file\n");
        exit(1);
    }else{
        init_file(mat1, size1, size1, fp1); 
    }



    fp2 = fopen(argv[2], "r");

    if(fp2 == NULL){
        printf("Error in opening the second file\n");
        exit(1);
    }else{
        init_file(mat2, size1, size1, fp2); 
    }
    

    struct Matrix *res = (struct Matrix*) malloc(sizeof(struct Matrix));
    res = init(res, mat1 -> row, mat1 -> col);
    // printf("res -> row = %d, res -> col = %d\n", res -> row, res -> col);
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // mulMatrix_kij(res, mat1, mat2);

    // mulMatrix_ikj(res, mat1, mat2);
    // mulMatrix_tradition(res, mat1, mat2);
    // mulMatrix_block(res, mat1, mat2);

    // mulMatrix_strassen(res, mat1, mat2, mat1 -> row);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mat1 -> row, mat2 -> col, mat1 -> col, 1.0f, 
        mat1 -> nums, mat1 -> col, mat2 -> nums, mat2 -> col, 0, res -> nums, res -> col);
    
    gettimeofday(&end, NULL);long timeuse =1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("The matrix multiplication used %lfs\n" , timeuse / 1000000.0);

    
    
    FILE *fout = fopen(argv[3], "w");
    if(fout == NULL){
        printf("Cannot create a file named: %s\n", argv[3]);
        exit(1);
    }
    int row = res -> row;
    int col = res -> col;
    char *str_res = (char*) malloc(10 * sizeof(char));
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            sprintf(str_res, "%.6f", res -> nums[i * col + j]);
            fputs(str_res, fout);
            fputc(' ', fout);
        }
        fputs("\n", fout);
    }
    
    fclose(fout);


    if(delMatrix(mat1)){
        printf("matrix1 has been successfully deleted\n");
    }

    if(delMatrix(mat2)){
        printf("matrix2 has been successfully deleted\n");
    };

    if(delMatrix(res)){
        printf("result matrix has been successfully deleted\n");
    }

    return 0;
}

