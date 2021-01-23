#include<iostream>
#include <sys/time.h>
/*
*created by qdl in 2020/11/23
*这是一个个人编写的时间测试类，用于测试一条程序的运行时间，时间以毫秒为单位
*/
class MyTime
{
private:
    struct timeval tvBegin, tvEnd;
public: 
    MyTime(){};
    void Start(){
        gettimeofday(&tvBegin, NULL);
    }
    void End(){
        gettimeofday(&tvEnd, NULL);
    }
    //获取毫米为单位的时间，精度为10^-6s
    double getTime(){               
        double dDuration = 1000 * (tvEnd.tv_sec - tvBegin.tv_sec) + ((tvEnd.tv_usec - tvBegin.tv_usec) / 1000.0);
        return dDuration;
    }
}myTime;