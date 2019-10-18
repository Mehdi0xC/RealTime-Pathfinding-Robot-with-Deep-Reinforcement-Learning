#include "config.hpp"
#include "nn.hpp"
#include "memory.hpp"
#include "utils.hpp"
#include "brain.hpp"
#include <iostream>

DQN::DQN()
{
        nStates  = Config:: nStates;
        nNeurons = Config:: nNeurons;
        nActions = Config:: nActions;
        nSamples = Config:: nSamples;
        Memory memory;
        MLP network;
        lastState = Utils::create2DArray(1,nStates);
        lastAction = Utils::create2DArrayOfIntegers(1,1);
        lastReward = Utils::create2DArray(1,1);
        action = 0;
        batchState = Utils::create2DArray(nSamples, nStates);
        batchNextState = Utils::create2DArray(nSamples, nStates);
        batchReward = Utils::create2DArray(nSamples, 1);
        batchAction = Utils::create2DArrayOfIntegers(nSamples, 1);
        randomIndices = Utils::create2DArrayOfIntegers(nSamples, 1);
}

int DQN::update(float** reward, float** newState)
{
    memory.push(lastState, newState, reward, lastAction);

    if(memory.counter < Config::learningIterations) 
        action = network.predict(newState, true);
    else
        action = network.predict(newState, false);

    if(memory.counter >= nSamples && memory.counter < Config::learningIterations) 
    {
        memory.generateRandomIndices(randomIndices);
        memory.sampleStates(batchState, randomIndices);
        memory.sampleNextStates(batchNextState, randomIndices);
        memory.sampleRewards(batchReward, randomIndices);
        memory.sampleActions(batchAction, randomIndices);
        network.learn(batchState, batchNextState, batchReward, batchAction);
    }
    
    lastAction[0][0] = action;
    Utils::copy(lastState,newState, 1, nStates);

    return action;
}
