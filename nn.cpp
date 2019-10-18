#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include "config.hpp"
#include "utils.hpp"
#include "nn.hpp"
using namespace std;

MLP::MLP()
{
    nStates = Config::nStates;
    nNeurons = Config::nNeurons;
    nActions = Config::nActions;
    nSamples = Config::nSamples;
    gamma = Config::gamma;
    learningRate = Config::learningRate;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 1);

    W1 = Utils::create2DArray(nStates,nNeurons);
    b1 = Utils::create2DArray(1,nNeurons);
    W2 = Utils::create2DArray(nNeurons,nActions);
    b2 = Utils::create2DArray(1,nActions);

    batchHiddenLayerOutput = Utils::create2DArray(nSamples,nNeurons);
    batchOutput = Utils::create2DArray(nSamples,nActions);
    batchHiddenLayerNextOutput = Utils::create2DArray(nSamples,nNeurons);
    error  = Utils::create2DArray(nSamples,nActions);
    maxIndices  = Utils::create2DArrayOfIntegers(nSamples,1);
    dW2 = Utils::create2DArray(nNeurons, nActions);
    dW1 = Utils::create2DArray(nStates, nNeurons);
    db2 = Utils::create2DArray(1, nActions);
    db1 = Utils::create2DArray(1, nNeurons);
    action = Utils::create2DArrayOfIntegers(1,1);

    dOutput = Utils::create2DArray(nSamples,nNeurons);

    batchNextOutput = Utils::create2DArray(nSamples,nActions);

    W2Transposed = Utils::create2DArray(nActions, nNeurons);

    batchHiddenLayerOutputTransposed = Utils::create2DArray(nNeurons, nSamples);
    batchStateTransposed = Utils::create2DArray(nStates, nSamples);
    hiddenLayerOutput = Utils::create2DArray(1, nNeurons);
    output = Utils::create2DArray(1, nActions);

    for(int i = 0 ; i < nStates ; i++)
        for(int j = 0 ; j < nNeurons ; j++)
            W1[i][j] = (1*d(gen))/sqrt(nStates+nNeurons);

    for(int i = 0 ; i < nNeurons ; i++)
        for(int j = 0 ; j < nActions ; j++)
            W2[i][j] = (1*d(gen))/sqrt(nNeurons+nActions);
}

void MLP::learn(float** batchState, float** batchNextState, float** batchReward, int** batchAction)
{


    Utils::clear(batchHiddenLayerOutput,nSamples,nNeurons);
    Utils::clear(batchHiddenLayerNextOutput,nSamples,nNeurons);
    Utils::clear(batchOutput,nSamples,nActions);
    Utils::clear(batchNextOutput,nSamples,nActions);
    Utils::clear(error ,nSamples,nActions);

    Utils::clear(dW2,nNeurons, nActions);
    Utils::clear(dW1,nStates, nNeurons);

    Utils::clear(db2,1, nActions);
    Utils::clear(db1,1, nNeurons);

    Utils::clear(dOutput,nSamples,nNeurons);

    Utils::clear(W2Transposed,nActions, nNeurons);
    Utils::clear(batchHiddenLayerOutputTransposed,nNeurons, nSamples);
    Utils::clear(batchStateTransposed,nStates, nSamples);
    Utils::clear(hiddenLayerOutput,1, nNeurons);
    Utils::clear(output,1, nActions);

    Utils::clearIntegers(maxIndices, nSamples, 1);


    Utils::dot(batchHiddenLayerOutput, batchState, W1, nSamples, nStates, nNeurons);
    Utils::partialSum(batchHiddenLayerOutput, b1, nSamples, nNeurons);
    Utils::relu(batchHiddenLayerOutput, nSamples,nNeurons);

    Utils::dot(batchOutput, batchHiddenLayerOutput, W2, nSamples, nNeurons, nActions);
    Utils::partialSum(batchOutput, b2, nSamples, nActions);

    Utils::copy(error, batchOutput, nSamples, nActions);

    /////////////////////////////////////////////

    Utils::dot(batchHiddenLayerNextOutput, batchNextState, W1, nSamples, nStates, nNeurons);
    Utils::partialSum(batchHiddenLayerNextOutput, b1, nSamples, nNeurons);
    Utils::relu(batchHiddenLayerNextOutput, nSamples,nNeurons);

    Utils::dot(batchNextOutput, batchHiddenLayerNextOutput, W2, nSamples, nNeurons, nActions);
    Utils::partialSum(batchNextOutput, b2, nSamples, nActions);

    /////////////////////////////////////////////
    Utils::argMax(maxIndices, batchNextOutput, nSamples, nActions, 1);

    for (int i = 0 ; i < nSamples ; i++)
        error[i][batchAction[i][0]] = gamma*batchNextOutput[i][maxIndices[i][0]] + batchReward[i][0];
    /////////////////////////////////////////////
    Utils::distance(error, batchOutput, nSamples, nActions);

    Utils::transpose(batchHiddenLayerOutputTransposed, batchHiddenLayerOutput, nSamples, nNeurons);
    Utils::dot(dW2, batchHiddenLayerOutputTransposed, error, nNeurons, nSamples, nActions);
    Utils::scalar(dW2, nNeurons, nActions, learningRate);
    Utils::sum(W2, dW2, nNeurons, nActions);
    //////////////////////////////////////////////
    Utils::sigma(db2, error, nSamples, nActions);
    Utils::scalar(db2, 1, nActions, learningRate);
    Utils::sum(b2, db2, 1, nActions);

    Utils::transpose(W2Transposed, W2, nNeurons, nActions);
    Utils::dot(dOutput, error, W2Transposed, nSamples, nActions, nNeurons);

    for(int i = 0; i < nSamples; i++)
        for(int j = 0; j < nNeurons; j++)
            if(batchHiddenLayerOutput[i][j] <= 0)
                dOutput[i][j] = 0;

    Utils::transpose(batchStateTransposed, batchState, nSamples, nStates);
    Utils::dot(dW1, batchStateTransposed, dOutput, nStates, nSamples, nNeurons);
    Utils::scalar(dW1, nStates, nNeurons, learningRate);
    Utils::sum(W1, dW1, nStates, nNeurons);
    //////////////////////////////////////////////


    Utils::sigma(db1, dOutput, nSamples, nNeurons);
    Utils::scalar(db1, 1, nNeurons, learningRate);
    Utils::sum(b1, db1, 1, nNeurons);
}

int MLP::predict(float** x, bool learning)
{
    Utils::clear(hiddenLayerOutput, 1, nNeurons);
    Utils::clear(output, 1, nActions);
    Utils::clearIntegers(action, 1, 1);

    Utils::dot(hiddenLayerOutput, x, W1, 1, nStates, nNeurons);
    Utils::sum(hiddenLayerOutput, b1, 1, nNeurons);
    Utils::relu(hiddenLayerOutput, 1, nNeurons);
    Utils::dot(output, hiddenLayerOutput, W2, 1, nNeurons, nActions);
    Utils::sum(output, b2, 1, nActions);

   if (learning == true)
    {

        mt19937 engine(device());
        uniform_int_distribution<int> r(0,100);


        Utils::softmax(output, output, 1, nActions);
        for(int i=1; i<nActions; i++)
            output[0][i] += output[0][i-1];

        int rawRandomNumber = r(engine);
        float randomNumber = float(rawRandomNumber)/100;
        //std::cout << randomNumber;

        for(int i=0; i<nActions; i++)
            if(randomNumber<output[0][i])
            {
                action[0][0] = i;
        		break;
            }
    }
    else
    {
        Utils::argMax(action, output, 1, nActions, 1);
    }
    return action[0][0];
}
