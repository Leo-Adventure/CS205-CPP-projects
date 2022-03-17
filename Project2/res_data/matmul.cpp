#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<sys/time.h>
#include<vector>
#include<iomanip>


using namespace std;
int main(int argc, char * argv[]){
    struct timeval total_t1, total_t2, read_t11, read_t12, read_t21, read_t22, t1, t2, write_t1, write_t2;
    gettimeofday(&total_t1, NULL);
    if(argc < 4){//在没有文件名输入的情况下报错并退出程序
        cout << "Three files needed." << endl;
        cerr << "Usage: " <<  argv[0] << " filename[s]\n";
        exit(EXIT_FAILURE);
    }
    

    //进行计时变量的初始化
    double time_use = 0.0, read_time_use = 0.0, total_time_use = 0.0;
    ifstream fin1;
    string str1 = argv[1];
    istringstream is(str1);
    string str_size;
    //从文件名当中获取矩阵的规模
    for(int i = 0; i < 2; i++){
        getline(is, str_size, '-');
    }
    getline(is, str_size, '.');

    int size = stoi(str_size);
    
    fin1.open(str1);
    //检测文件1是否能正常打开
    if(!fin1.is_open()){
        cerr << "Could not open " << str1 << endl;
        fin1.clear();
        exit(EXIT_FAILURE);
    }else{
        cout << "Access file1 successfully" << endl;
    }
    //创建并初始化定长数组
    double ** mat1 = new double*[size];
    double ** mat2 = new double*[size];
    double ** res = new double*[size];
    for(int i = 0; i < size; i++)
	{
		mat1[i] = new double[size];
        mat2[i] = new double[size];
        res[i] = new double[size];
	}
    gettimeofday(&read_t11, NULL);//在文件流创建后，在文件开始读取前，记录起始时间
    int line = 0;
    while(!fin1.eof()){
        for(int i = 0; i < size; i++){
            fin1 >> mat1[line][i];
        }
        line++;  
    }
    gettimeofday(&read_t12, NULL);//在文件流关闭之前，在文件读取结束之后，记录结束时间
    fin1.close();

    //将读取文件一所用的时间加入文件读取的总时间当中
    read_time_use += (read_t12.tv_sec - read_t11.tv_sec) + (double)(read_t12.tv_usec - read_t11.tv_usec) / 1000000.0; 
    
    ifstream fin2;

    string str2 = argv[2];
   
    fin2.open(str2);

    if(!fin2.is_open()){
        cerr << "Could not open " << str2 << endl;
        fin2.clear();
        exit(EXIT_FAILURE);
    }else{
        cout << "Access file2 successfully" << endl;
    }

   
    gettimeofday(&read_t21, NULL);
    int line2 = 0;
    while(!fin2.eof()){
        for(int i = 0; i < size; i++){
            fin2 >> mat2[line2][i];
        }
        line2++;  
    }
    gettimeofday(&read_t22, NULL);
    fin2.close();

    read_time_use += (read_t22.tv_sec - read_t21.tv_sec) + (double)(read_t22.tv_usec - read_t21.tv_usec) / 1000000.0; 
    
   
    gettimeofday(&t1, NULL);
    //进行矩阵的运算
    for(int k = 0; k < size; k++){
        for(int i = 0; i < size; i++){
            double r = mat1[i][k];
            for(int j = 0; j < size; j++){
                res[i][j] += r * mat2[k][j];
            }
        }
    }
    gettimeofday(&t2, NULL);
    

    time_use = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;

    cout << "The reading process used: " << read_time_use << "s" << endl;
    cout << "The computation used: " << time_use << "s" << endl;

    ofstream fout(argv[3]);
    
    //设置文件格式为科学计数法
    fout.setf(ios_base::scientific, ios_base::floatfield);
    gettimeofday(&write_t1, NULL);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            fout << res[i][j] << " ";
        }
        fout << endl;
    }
    gettimeofday(&write_t2, NULL);
    fout.close();
    //在完成了数组的使用之后释放内存
    delete[] mat1;
    delete[] mat2;
    delete[] res;
    //计算写入文件的时间
    double write_time_use = (write_t2.tv_sec - write_t1.tv_sec) + (double)(write_t2.tv_usec - write_t1.tv_usec) / 1000000.0;
    cout << "The writing process used: " << write_time_use << "s" << endl;
    
    gettimeofday(&total_t2, NULL);
    //程序的总用时
    total_time_use = (total_t2.tv_sec - total_t1.tv_sec) + (double)(total_t2.tv_usec - total_t1.tv_usec) / 1000000.0;
    cout << "The total time used in the program is: " << total_time_use << "s" << endl;
    //计算出文件IO时间占总时间的比例
    double rate = (read_time_use + write_time_use) / total_time_use * 100.0;
    cout << "The IO time take up " << rate << "%" << "of the total time" << endl;
    return 0;

}
