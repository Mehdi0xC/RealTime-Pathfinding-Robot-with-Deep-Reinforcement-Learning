#ifndef BRAIN_H_
#define BRAIN_H_
#include "memory.hpp"
#include "nn.hpp"
class DQN 
{ 
    // Access specifier 
    public:
        DQN();
        int nStates;
        int nNeurons;
        int nActions;
        int nSamples;
        Memory memory;
        MLP network;
        float** lastState;
        int** lastAction;
        float** lastReward;
        int action;
        float** batchState;
        float** batchNextState;
        float** batchReward;
        int** batchAction;
        int** randomIndices;
        int update(float** reward, float** newState);

}; 
#endif
