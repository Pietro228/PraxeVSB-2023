//
// Created by user on 5/28/2023.
//
#include "vector"
#include "iostream"
#include "math.h"

using namespace std;

#ifndef NEAURALNETWORK_SHAPE_RECOGNITION_LAYER_H
#define NEAURALNETWORK_SHAPE_RECOGNITION_LAYER_H


class Layer {
public:
    int layerSize;

    vector<double> neurons;//after activation value
    vector<double> calculatedNeurons;//weighted without passing activation function
    vector<double> inputNeurons;

    vector<vector<double>> weights; // for one output multiple inputs
    vector<vector<double>> costGradientW;

    vector<double> biases;
    vector<double> costGradientB;

    void InitializeRandomWeights();
    void ApplyGradient(double learnRate);
    double NeuronCost(double output, double expectedOutput);
    double NeuronCostDerivative(double output, double expectedOutput);
    vector<double> CalculateOutputNeuronValues(vector<double> expectedOutputs);
    void UpdateGradients(vector<double> neuronValues);
    vector<double> CalculateHiddenLayerNeuronValues(Layer oldLayer, vector<double> oldNeuronValues);


    Layer(int layerSize, vector<double> &inputNeurons){
        this->layerSize = layerSize;
        this->neurons.resize(layerSize,0);
        this->inputNeurons = inputNeurons;
        calculatedNeurons.resize(layerSize,0);
        costGradientW.resize(layerSize, vector<double>(inputNeurons.size(),0));
        costGradientB.resize(layerSize);

        weights.resize(layerSize,vector<double>(inputNeurons.size(),1));
        biases.resize(layerSize, 0);
        InitializeRandomWeights();
    }

    void CalculateNeurons(){
        for (int i = 0; i < layerSize; ++i) {
            double calculatedNeuron = biases[i];
            for (int j = 0; j < inputNeurons.size(); ++j) {
                calculatedNeuron += inputNeurons[j] * weights[i][j];
            }
            calculatedNeurons[i] = calculatedNeuron;
            neurons[i] = Activation(calculatedNeuron);
            //neurons[i] = max(0.0, calculatedNeuron);
        }
    }

    double Activation(double calculatedNeuron){
        return 1 / (1 + exp(-(calculatedNeuron)));
    }

    double ActivationDerivative(double calculatedNeuron){
        double activation = Activation(calculatedNeuron);
        return activation * (1 - activation);
    }
};


#endif //NEAURALNETWORK_SHAPE_RECOGNITION_LAYER_H
