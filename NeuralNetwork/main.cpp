#include <iostream>
#include "NeuralNetwork.h"
#include "Layer.h"

using namespace std;

int main() {
    DataPoint d1({1,0},{0,1});
    DataPoint d2({0,1},{1,0});
    DataPoint d3({0,0},{1,1});
    DataPoint d4({1,1},{0,0});

    NeuralNetwork network({2,3,2});
    for (int i = 0; i < 100000000; ++i) {
        network.Learn({d1,d2,d3,d4},0.001);
        if(network.Cost(d1) < 0.000000000001)
            break;
        vector<double> output = network.CalculateOutputs(d1.inputs);
    }
    vector<double> output = network.CalculateOutputs(d1.inputs);
    cout<<endl<<endl<<endl;
    cout<<output[0]<<endl;
    cout<<output[1]<<endl;
    cout<<network.Cost(d1)<<endl;
    return 0;
}
