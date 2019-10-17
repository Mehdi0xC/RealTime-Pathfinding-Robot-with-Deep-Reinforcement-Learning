#include <iostream>
#include <string>
#include "config.hpp"
#include "utils.hpp"
#include "brain.hpp"
using namespace std;

int main()
{
    string command;
    DQN agent;
    float ** state = Utils::create2DArray(1,Config::nStates);
    float ** reward = Utils::create2DArray(1, 1);
    while(true)
    {
        cin >> command;
        Utils::stateDecoder(state, command);
        cin >> command;
        Utils::rewarder(reward, command);
        cout << agent.update(reward, state) << endl;
    }    
}
