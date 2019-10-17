#ifndef MEMORY_H__
#define MEMORY_H__
#include <random>

class Memory 
{
    public:
        Memory();
        void generateRandomIndices(int** result);
        void sampleStates(float** result, int** indices);
        void sampleNextStates(float** result, int** indices);
        void sampleRewards(float** result, int** indices);
        void sampleActions(int** result, int** indices);
        void push(float** state, float** nextState, float** reward, int** action);
        std::random_device device;     // only used once to initialise (seed) engine
    	float  ** states;
    	float  ** nextStates;
    	float  ** rewards;
    	int  ** actions;
        int counter;
        int size;
        int nSamples;
        int nNeurons;
        int nActions;
        int nStates;
        bool full;
};

#endif
