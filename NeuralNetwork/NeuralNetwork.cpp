//
// Created by user on 5/28/2023.
//

#include "NeuralNetwork.h"
#include "random"

vector<double> NeuralNetwork::CalculateOutputs(vector<double> inputs){
    inputLayer = inputs;
    for (int layer = 0; layer < layers.size(); ++layer) {
        if(layer == 0){
            layers[layer].inputNeurons = inputLayer;
        } else{
            layers[layer].inputNeurons = layers[layer - 1].neurons;
        }
        layers[layer].CalculateNeurons();
    }
    return layers[layers.size()-1].neurons;
}

vector<double> NeuralNetwork::GetOutputs(){

}

void NeuralNetwork::SetData(vector<DataPoint> data){
    this->data = data;
}

double NeuralNetwork::Cost(DataPoint dataPoint){
    vector<double> outputs = CalculateOutputs(dataPoint.inputs);
    Layer outputLayer = layers[layers.size() - 1];
    double cost = 0;

    for (int neuron = 0; neuron < outputs.size(); ++neuron) {
        cost += outputLayer.NeuronCost(outputs[neuron], dataPoint.expectedOutputs[neuron]);
    }
    return cost;
}

double NeuralNetwork::Cost(vector<DataPoint> data){
    double totalCost = 0;

    for (int dataPoint = 0; dataPoint < data.size(); ++dataPoint) {
        totalCost += Cost(data[dataPoint]);
    }
    return totalCost / data.size();
}

void NeuralNetwork::Learn(vector<DataPoint> trainingData, double learnRate) {
    for (int i = 0; i < trainingData.size(); ++i) {
        UpdateAllGradient(trainingData[i]);
    }

    for (int i = 0; i < layers.size(); ++i) {
        layers[i].ApplyGradient(learnRate);
    }
}

void NeuralNetwork::UpdateAllGradient(DataPoint dataPoint){
    CalculateOutputs(dataPoint.inputs);

    vector<double> neuronValues = layers[layers.size() - 1].CalculateOutputNeuronValues(dataPoint.expectedOutputs);
    layers[layers.size() - 1].UpdateGradients(neuronValues);


    for (int hiddenLayerIndex = layers.size() - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
        neuronValues = layers[hiddenLayerIndex].CalculateHiddenLayerNeuronValues(layers[hiddenLayerIndex + 1], neuronValues);
        layers[hiddenLayerIndex].UpdateGradients(neuronValues);
    }
}

///old learn slow as fuck by miracle worked tho
/*
 * const double h = 0.000001;
    double originalCost = Cost(trainingData);

    for (int layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        for (int weightArrayIndex = 0; weightArrayIndex < layers[layerIndex].weights.size(); ++weightArrayIndex) {
            for (int weightIndex = 0; weightIndex < layers[layerIndex].weights[weightArrayIndex].size(); ++weightIndex) {
                layers[layerIndex].weights[weightArrayIndex][weightIndex] += h;
                double deltaCost = Cost(trainingData) - originalCost;
                layers[layerIndex].weights[weightArrayIndex][weightIndex] -= h;
                layers[layerIndex].costGradientW[weightArrayIndex][weightIndex] = deltaCost / h;
            }
        }

        for (int biasIndex = 0; biasIndex < layers[layerIndex].biases.size(); ++biasIndex) {
            layers[layerIndex].biases[biasIndex] += h;
            double deltaCost = Cost(trainingData) - originalCost;
            layers[layerIndex].biases[biasIndex] -= h;
            layers[layerIndex].costGradientB[biasIndex] = deltaCost / h;
        }
    }
    for (int layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        layers[layerIndex].ApplyGradient(learnRate);
    }
 */