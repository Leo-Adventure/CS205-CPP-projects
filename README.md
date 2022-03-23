# CS205-C/C++-projects

## Integer Multiplication

A high accuracy Integer Multiplication is implemented in this project.

## Matrix Multiplication

A matrix multiplication using some improvements is implemented in this project.

### Requirement

Please implement a program to multiply two matrices in two files. 

There are 6 files in the attachment. mat-A-32.txt is for matrix A, the matrix size is 32x32. Other files are similar. 

1. When you run the program as follows, it will output the result into an output file such as `out32.txt`  
 
   $./matmul mat-A-32.txt mat-B-32.txt out32.txt $.

   $./matmul mat-A-256.txt mat-B-256.txt out256.txt $

   $./matmul mat-A-2048.txt mat-B-2048.txt out2048.txt $

2. Please implement the matrix multiplication in float and double separately, and compare the time consumed and the results. Give detailed analysis on the speed and accuracy. 

3. You can try to improve the speed, and introduce the methods in the report.

## Matrix Structure and Multiplication in C

A basic matrix structure and its multiplication is implemented in C language in this project.

### Requirements 

1. The programming language can only be C, not C++. Please save your source code into *.c files, and compile them using a C compiler such as gcc (not g++). Try to use Makefile or CMake to manage your source code. 

2. Design a struct for matrices, and the struct should contain the data of a matrix, the number of columns, the number of rows, etc. 

3. Implement some functions to 

   ◦ create a matrix 

   ◦ delete a matrix 

   ◦ copy a matrix (copy the data from a matrix to another) 

   ◦ multiply two matrices ◦ some other functions needed 

4. When you run the program as follows, it will output the result into an output file such as out32.txt . The data files are the same with those in Project 2. $./matmul mat-A-32.txt mat-B-32.txt out32.txt $

   $./matmul mat-A-256.txt mat-B-256.txt out256.txt $

   $./matmul mat-A-2048.txt mat-B-2048.txt out2048.txt 5.$ 

   Try to improve the speed of matrix multiplication. Introduce how you improve it in the report. You should explicitly introduce the differences between this one and Project 2. 6. Compare the speed of your implementation with OpenBLAS (https://www.openblas.net/)

## A Class for Matrices

A  complete class of matrix is implemented taking OpenCV as example is implemented in this project

### Requirements 

1. Design a class for matrices, and the class should contain the data of a matrix and related information such the number of rows, the number of columns, the number of channels, etc.
2. The class support different data types. It means that the matrix elements can be unsigned char , short , int , float , double , etc. 
3. Do not use memory hard copy if a matrix object is assigned to another. Please carefully handle the memory management to avoid memory leak and to release memory multiple times. 
4. Implement some frequently used operators including but not limit to = , == , + , - , * , etc. Surely the matrix multiplication in Project 3 should be included.
5. Implement region of interest (ROI) to avoid memory hard copy. 
6. Test your program on X86 and ARM platforms, and describe the differences. 
7. Class cv::Mat is a good example for this project. https://docs.opencv.org/ master/d3/d63/classcv_1_1Mat.htm

## A Simple CNN Model

A simple CNN model is implemented in this project.

### Requirement: Please implement a simple convolutional neural network (CNN) model using C/C++. 

You can follow https://github.com/ShiqiYu/SimpleCNNbyCPP where a pretrained CNN model is provided. The model contains 3 convolutional layers and 1 fully connected layer. The model can predict if the input image is a person (upper body only) or not. More details about the model can be found at SimpleCNNbyCPP web site. 

You are welcome to implement more CNN layers, and to make the implemented CNN to be more general (such as the convolutional layer can be for any size of kernels, not just 3x3). 

Do not use any third-party library except OpenCV to read image data from image files. You should implement all CNN layers using C/C++. 

#### Hints

1. Only the forward part is required to implement, and the backward part (the training part) is not mandatory. 
2. You can implement an unoptimized version firstly to verify the correctness of the implementation. Then you can optimize it for a better speed. Make your source code simple, beautiful and efficient. 
3.  The convolutional operation can be implemented by matrix multiplication. Do not waste your experience and the source code in Project 2, Project 3 and Project 4. 
4. The parameters trained have been put into a CPP file /weights/face_binary_cls.cpp . You can just include it into your project. 
5. You can use OpenCV to read images. The image data stored in cv::Mat is unsigned char type. You should convert the data to float and normalize it to range [0.0, 1.0] firstly before operate it in a convolutional layer. Be careful with the order of pixel colors (channels) in cv::Mat. It is BRG, not RGB. The input of the CNN model should be a 3x128x128 (channel, height, width) data blob (matrix). You can adapt your class in Project 4 for this project. 
6. The output of the CNN model is a vector with two float numbers [c0, c1] . c0 + c1 = 1.0 and c1 is the possibility that the input is a person (upper body only). 
7. Test your program on X86 and ARM platforms, and describe the differences. Be careful that char is different on X86 and ARM. It is signed char on one, and unsigned char on another. 



### Highlight

In this project 5, I paid a lot of efforts in it and I was honorly selected as an example report by my Prof.

The link is https://github.com/ShiqiYu/CPP/blob/main/projects/Project5-goodexamples-2021fall-by%E5%BB%96%E9%93%AD%E9%AA%9E.pdf

## Thinking 

I am very grateful for what this course has taught me. This course is one of the best computer courses in Southern University of Science and Technology, and I believe it will be one of the best computer courses in China in the future. I am honored to be a C++ teaching assistant through my hard work this semester. In the future study career, I will continue to improve my C++ level.
