#ifndef UTILS_H_
#define UTILS_H_
#include <string>
class Utils 
{ 
    public: 
        static void    stateDecoder(float ** state, std::string command);
        static float** create2DArray(int r, int c);
        static void    clear(float** x, int r, int c);
        static void    clearIntegers(int** x, int r, int c);
        static int**   create2DArrayOfIntegers(int r, int c);
        static void    dot(float** result, float ** x, float ** y, int m, int n, int p);
        static void    sum(float ** x, float ** y, int r, int c);
        static void    copy(float ** x, float ** y, int r, int c);
        static void    distance(float** result, float ** x,  int r, int c);
        static void    relu(float ** x, int r, int c);
        static void    scalar(float ** x, int r, int c , float alpha);
        static void    argMax(int** result, float **x, int r, int c, int axis);
        static void    transpose(float ** result, float ** x, int r, int c);
        static void    sigma(float** result, float ** x, int r, int c);
        static void    partialSum(float** result,  float** px, int r, int c);
        static void    rewarder(float** reward, std::string command);
}; 

#endif
