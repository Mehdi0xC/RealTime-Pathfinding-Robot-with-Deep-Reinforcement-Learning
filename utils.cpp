#include <iostream>
#include <string>
#include "config.hpp"
#include "utils.hpp"
using namespace std;

// FUNCTION FOR DECODING STATES INCOMING FROM UART COMMANDS
void Utils::stateDecoder(float ** state, string command)
{
for (int i=0; i<=Config::nStates; i++)
    if(command[i] == '1')
        state[0][i] = 1;
    else
        state[0][i] = 0;
}

// FUNCTION FOR CREATING AND INITIALIZING A 2D POINTER
float** Utils::create2DArray(int r, int c)
{
    float** array2D = 0;
    array2D = new float*[r];
    for (int h = 0; h < r; h++)
    {
        array2D[h] = new float[c];
        for (int w = 0; w < c; w++)
        {
                array2D[h][w] = 0;
        }
    }
    return array2D;
}

// FUNCTION FOR CLEARING VALUES OF A 2D POINTER OF FLOATING POINTS
void Utils::clear(float** x, int r, int c)
{
    for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
            x[i][j] = 0;
}

void Utils::clearIntegers(int** x, int r, int c)
{
    for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
            x[i][j] = 0;
}


// FUNCTION FOR CLEARING VALUES OF A 2D POINTER OF INTEGERS
int** Utils::create2DArrayOfIntegers(int r, int c)
{
    int** result  = 0;
    result = new int*[r];
    for (int i = 0; i < r; i++)
    {
        result[i] = new int[c];

        for (int j = 0; j < c; j++)
        {
                result[i][j] = 0;
        }
    }
    return result;
}

// FUNCTION FOR CALCULATING DOT PRODUCT OF MATRICES
void  Utils::dot(float** result, float ** x, float ** y, int m, int n, int p)
{
	Utils::clear(result, m, p);
    for(int i = 0 ; i < m ; i++)
        for(int j = 0 ; j < p ; j++)
            for(int k = 0 ; k < n ; k++)
                {
                  result[i][j] +=  x[i][k] * y[k][j];
                }
}

// FUNCTION FOR CALCULATING SUM OF MATRICES
void  Utils::sum(float ** x, float ** y, int r, int c)
{
    for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
            x[i][j] +=  y[i][j];
}

// FUNCTION FOR COPYING  MATRICES
void  Utils::copy(float ** x, float ** y, int r, int c)
{
    for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
            x[i][j] =  y[i][j];
}

// FUNCTION FOR CALCULATING DISTANCE OF MATRICES
void  Utils::distance(float ** result, float ** x, int r, int c)
{
    for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
            result[i][j] = result[i][j] - x[i][j];
}

// FUNCTION FOR CALCULATING DISTANCE OF MATRICES
void Utils::relu(float ** x, int r, int c)
{
        for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
        if(x[i][j] <= 0)
            x[i][j] = 0;
}

// FUNCTION FOR CALCULATING SCALAR MULTIPLICATION ON MATRICES
void Utils::scalar(float ** x, int r, int c , float alpha)
{
        for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
            x[i][j] = alpha*x[i][j];
}

// FUNCTION FOR CALCULATING ARGMAX OF MATRICES
void Utils::argMax(int** result, float **x, int r, int c, int axis)
{
    if(axis==0)
    {
    for(int j = 0 ; j < c ; j++)
    {
        float max = x[0][j];
        for(int i = 1 ; i < r ; i++)
        if(x[i][j] > max)
        {
            max = x[i][j];
            result[0][j] = i;
        }    
    } 
    }
    else
    {
    for(int i = 0 ; i < r ; i++)
    {
        float max = x[i][0];
        for(int j = 1 ; j < c ; j++)
        if(x[i][j] > max)
        {
            max = x[i][j];
            result[i][0] = j;
        }    
    } 
    }
}

// FUNCTION FOR CALCULATING TRANSPOSE OF MATRICES
void Utils::transpose(float ** result, float ** x, int r, int c)
{
        for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
            result[j][i] = x[i][j];
}

// FUNCTION FOR CALCULATING SUM OF MATRIX ROWS OR COLUMNS
void Utils::sigma(float** result, float ** x, int r, int c)
{
        for(int j = 0 ; j < c ; j++)
        for(int i = 0 ; i < r ; i++)
        result[0][j] += x[i][j];
}

// FUNCTION FOR CALCULATING SOFTMAX FUNCTION
void Utils::softmax(float** result, float ** x, int r, int c, float temperature)
{
        for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
        result[i][j] = exp(x[i][j]/Config::softmaxTemperature);

        float sumExponents = 0;
        for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
        sumExponents += result[i][j];

        for(int i = 0 ; i < r ; i++)
        for(int j = 0 ; j < c ; j++)
        result[i][j] = result[i][j]/ sumExponents;

}

def softmax(A):
    expA = np.exp(A/learningCoreSettings["softmaxTemperature"])
    return expA / expA.sum()


// FUNCTION FOR CALCULATING PARTIAL SUM
void Utils::partialSum(float** result,  float** px, int r, int c)
{
	for(int i = 0 ; i < r ; i++)
	for(int j = 0 ; j < c ; j++)
	result[i][j] +=  px[0][j];
}

void Utils::rewarder(float** reward, string command)
{
    if(command == "1")
        reward[0][0] = 1;
    else
        reward[0][0] = -1;
}
