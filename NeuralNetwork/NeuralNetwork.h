//
// Created by user on 5/28/2023.
//
#include "Layer.h"


#ifndef NEAURALNETWORK_SHAPE_RECOGNITION_NEURALNETWORK_H
#define NEAURALNETWORK_SHAPE_RECOGNITION_NEURALNETWORK_H

struct DataPoint{
public:
    vector<double> inputs;
    vector<double> expectedOutputs;
    DataPoint(vector<double> inputs, vector<double> expectedOutputs){
        this->inputs = inputs;
        this->expectedOutputs = expectedOutputs;
    }
};

class NeuralNetwork {
public:
    vector<Layer> layers;
    vector<DataPoint> data;
    vector<double> inputLayer;
    vector<double> expectedOutputs;

    vector<double> CalculateOutputs(vector<double> inputs);
    vector<double> GetOutputs();
    void SetData(vector<DataPoint> data);
    double Cost(DataPoint dataPoint);
    double Cost(vector<DataPoint> data);
    void Learn(vector<DataPoint> trainingData, double learnRate);
    void UpdateAllGradient(DataPoint dataPoint);


    NeuralNetwork(vector<int> layerSizes){
        inputLayer.resize(layerSizes[0],1);
        expectedOutputs.resize(layerSizes[layerSizes.size() - 1],0);
        for (int i = 0; i < layerSizes.size() - 1; ++i) {
            if(i == 0){
                layers.emplace_back(layerSizes[i + 1],inputLayer);
            } else{
                layers.emplace_back(layerSizes[i + 1],layers[i - 1].neurons);
            }
        }
    }
};


#endif //NEAURALNETWORK_SHAPE_RECOGNITION_NEURALNETWORK_H
