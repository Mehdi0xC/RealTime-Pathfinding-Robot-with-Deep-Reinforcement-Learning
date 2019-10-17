#ifndef CONFIG_H_
#define CONFIG_H_

class Config 
{ 
    public: 
    const static int nStates = 3;
    const static int nActions = 3;
    const static int nNeurons = 10;
    constexpr static float gamma = 0.9;
    constexpr static float learningRate = 0.001;
    const static int nSamples = 4;
    const static int memoryCapacity = 10000;
    const static int softmaxTemperature = 10;
    const static int learningIterations = 2500;
}; 

#endif
