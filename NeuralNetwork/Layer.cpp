//
// Created by user on 5/28/2023.
//

#include "Layer.h"
#include "random"

std::default_random_engine rnd{std::random_device{}()};
std::uniform_real_distribution<double> dist(-1, 1);

void Layer::InitializeRandomWeights(){
    for (int neuron = 0; neuron < layerSize; ++neuron) {
        for (int weightArray = 0; weightArray < weights.size(); ++weightArray) {
            for (int weight = 0; weight < weights[weightArray].size(); ++weight) {
                weights[weightArray][weight] = dist(rnd);
            }
        }
    }
}

void Layer::ApplyGradient(double learnRate) {
    for (int neuronIndex = 0; neuronIndex < layerSize; ++neuronIndex) {
        biases[neuronIndex] -= costGradientB[neuronIndex] * learnRate;
        for (int weightIndex = 0; weightIndex < weights[neuronIndex].size(); ++weightIndex) {
            weights[neuronIndex][weightIndex] -= costGradientW[neuronIndex][weightIndex] * learnRate;
        }
    }
}

double Layer::NeuronCost(double output, double expectedOutput){
    double error = output - expectedOutput;
    return error * error;
}

double Layer::NeuronCostDerivative(double output, double expectedOutput){
    return 2*(output - expectedOutput);
}

vector<double> Layer::CalculateOutputNeuronValues(vector<double> expectedOutputs){
    vector<double> neuronValues(expectedOutputs.size());

    for (int i = 0; i < neuronValues.size(); ++i) {
        double costDerivative = NeuronCostDerivative(neurons[i], expectedOutputs[i]);
        double activationDerivative = ActivationDerivative(calculatedNeurons[i]);
        neuronValues[i] = activationDerivative * costDerivative;
    }

    return neuronValues;
}

void Layer::UpdateGradients(vector<double> neuronValues){
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[i].size(); ++j) {
            double derivativeCostWrtWeight = inputNeurons[j] * neuronValues[i];

            costGradientW[i][j] += derivativeCostWrtWeight;
        }

        double derivativeCostWrtBias = 1 * neuronValues[i];
        costGradientB[i] += derivativeCostWrtBias;
    }
}

vector<double> Layer::CalculateHiddenLayerNeuronValues(Layer oldLayer, vector<double> oldNeuronValues){
    vector<double> newNeuronValues(layerSize);

    for (int newNeuronIndex = 0; newNeuronIndex < newNeuronValues.size(); ++newNeuronIndex) {
        double newNeuronValue = 0;

        for (int oldNeuronIndex = 0; oldNeuronIndex < oldNeuronValues.size(); ++oldNeuronIndex) {
            double calculatedNeuronDerivative = oldLayer.weights[oldNeuronIndex][newNeuronIndex];
            newNeuronValue += calculatedNeuronDerivative * oldNeuronValues[oldNeuronIndex];
        }
        newNeuronValue *= ActivationDerivative(calculatedNeurons[newNeuronIndex]);
        newNeuronValues[newNeuronIndex] = newNeuronValue;
    }

    return newNeuronValues;
}