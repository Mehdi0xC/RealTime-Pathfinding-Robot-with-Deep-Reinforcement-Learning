#include <iostream>
#include <random>
 #include "config.hpp"
 #include "utils.hpp"
 #include "memory.hpp"
using namespace std;

Memory::Memory()
{
        counter = 0;
        size = Config::memoryCapacity;
        nSamples = Config::nSamples;
        nNeurons = Config::nNeurons;
        nActions = Config::nActions;
        nStates = Config::nStates;
        full = false;
        states = Utils::create2DArray(size, nStates);
    	nextStates = Utils::create2DArray(size, nStates);
    	rewards = Utils::create2DArray(size, 1);
    	actions = Utils::create2DArrayOfIntegers(size, 1);  
}

void Memory::push(float** state, float** nextState, float** reward, int** action)
{
    if (counter == size)
    {
        full = true;
        counter = 0;
    }
    for (int i = 0 ; i < nStates ; i++)
    {
        states[counter][i] = state[0][i];
        nextStates[counter][i] = nextState[0][i];
    }
    rewards[counter][0] = reward[0][0];
    actions[counter][0] = action[0][0];
    counter++;
}

void Memory::generateRandomIndices(int** result)
{
    mt19937 engine(device());
    if(full == false)
    {
    uniform_int_distribution<int> r(0,counter-1);
    for (int i = 0 ; i < nSamples ; i++)
        result[i][0] = r(engine);
    }
    else
    {
    uniform_int_distribution<int> r(0,size-1);
    for (int i = 0 ; i < nSamples ; i++)
        result[i][0] = r(engine);        
    }
}

void Memory::sampleStates(float** result, int** indices)
{
    for (int i = 0 ; i < nSamples ; i++)
    for (int j = 0 ; j < nStates ; j++)
        result[i][j] = states[indices[i][0]][j];  
}

void Memory::sampleNextStates(float** result, int** indices)
{
    for (int i = 0 ; i < nSamples ; i++)
    for (int j = 0 ; j < nStates ; j++)
        result[i][j] = nextStates[indices[i][0]][j];
}

void Memory::sampleRewards(float** result, int** indices)
{
    for (int i = 0 ; i < nSamples ; i++)
        result[i][0] = rewards[indices[i][0]][0];
}

void Memory::sampleActions(int** result, int** indices)
{
    for (int i = 0 ; i < nSamples ; i++)
        result[i][0] = actions[indices[i][0]][0];
}
