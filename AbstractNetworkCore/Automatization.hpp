#pragma once
#include <functional>

namespace mnn {
	class AbstractNeuron;
	class AbstractLayerNetwork;

	/*
	The enum defining type of connection of the neurons in the network.
	
	Variants of connection:
	* NoDefaultConnection - no neuron will be connected to another one.
	* EachFrompreviousLayerWithoutBias - Simple connects every neuron to each neurn in the previous layer.
	* EachFromPreviousLayerWithBias - Similar type of connection. The only difference is an additional connection to constant value(1.0) neuron, called Bias, for each of the neurons.
	*/
	enum class ConnectionPattern {
		NoDefaultConnection, EachFromPreviousLayerWithBias, EachFromPreviousLayerWithoutBias,
	};

	/*
	The function generates and returns a pointer to a NeuralNetwork with *input_number* inputs, *output_number* outputs, *hidden_layers_number* hidden layers with *neurons_per_hidden_layer* neurons in each,
	The connection type is determined by *connection*. Default vriant connects a neuron to all the neurons from previous layer and to a bias neuron.
	For more specific details see ConnectionPattern enum.
	
	WeightFunction is an lambda-function-argument which takes the arguments:
	* The neuron is being linked to
	* The neuron is being linked
	It returns a float -> weight of the Link.
	
	By default, the weights of all Links are equal to 1.0
	
	*eta* - coefficient determining overall net learning rate. Default value - 0.15
	*alpha* - coefficient determining overall net momentum. Default value - 0.5
	*/
	AbstractLayerNetwork* generateTypicalLayerNeuralNetwork(size_t inputs_number, size_t outputs_number,
															size_t hidden_layers_number, size_t neurons_per_hidden_layer,
															ConnectionPattern connection = ConnectionPattern::EachFromPreviousLayerWithBias,
															std::function<float(AbstractNeuron*, AbstractNeuron*)> weightFunction = 
																[](AbstractNeuron* neuron, AbstractNeuron* input) -> float {
																	return 1.f;
																}, float eta = 0.15f, float alpha = 0.5f);
}