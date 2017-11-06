#pragma once
#include "Shared.hpp"
#include <functional>
namespace mnn {
	class AbstractNeuron;
	class AbstractBackpropagationNeuron;
	class AbstractLayerNetwork;
	class AbstractBackpropagationLayerNetwork;
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
	Default weight function for the typical automatically created AbstractLayerNeuralNetwork.
	As of version 0.1.40, it makes all the weights equal to +1.f;
	*/
	float default_weights(AbstractNeuron const& neuron, AbstractNeuron const& input);
	/*
	Analog for default weight function.
	It makes all the weight equal to a random number between -1.f and +1.f;
	*/
	float random_weights(AbstractNeuron const& neuron, AbstractNeuron const& input);

	/*
	The function generates and returns a pointer to a NeuralNetwork with *input_number* inputs, *output_number* outputs, *hidden_layers_number* hidden layers with *neurons_per_hidden_layer* neurons in each,
	The connection type is determined by *connection*. Default variant connects a neuron to all the neurons from previous layer and to a bias neuron.
	For more specific details see ConnectionPattern enum.
	
	WeightFunction is an lambda-function-argument which takes the arguments:
	* The neuron is being linked to
	* The neuron is being linked
	It returns a float -> weight of the Link.
	
	By default, the weights of all Links are equal to 1.0
	*/
	AbstractLayerNetwork* generateTypicalLayerNeuralNetwork(size_t inputs_number, size_t outputs_number,
															size_t hidden_layers_number, size_t neurons_per_hidden_layer,
															ConnectionPattern connection = ConnectionPattern::EachFromPreviousLayerWithBias,
															std::function<Type(AbstractNeuron const&, AbstractNeuron const&)> weightFunction = default_weights);

	/*
	The function generates and returns a pointer to a NeuralNetwork with *input_number* inputs, *output_number* outputs, *hidden_layers_number* hidden layers with *neurons_per_hidden_layer* neurons in each,
	The connection type is determined by *connection*. Default variant connects a neuron to all the neurons from previous layer and to a bias neuron.
	For more specific details see ConnectionPattern enum.

	Neurons are prepared for learningProcess(backpropagation) call in order to learn.

	WeightFunction is an lambda-function-argument which takes the arguments:
	* The neuron is being linked to
	* The neuron is being linked
	It returns a float -> weight of the Link.

	By default, the weights of all Links are equal to 1.0

	*eta* - coefficient determining overall net learning rate. Default value - 0.15
	*alpha* - coefficient determining overall net momentum. Default value - 0.5
	*/
	AbstractBackpropagationLayerNetwork* generateTypicalBackpropagationLayerNeuralNetwork(size_t inputs_number, size_t outputs_number,
																		   size_t hidden_layers_number, size_t neurons_per_hidden_layer,
																		   ConnectionPattern connection = ConnectionPattern::EachFromPreviousLayerWithBias,
																		   std::function<Type(AbstractNeuron const&, AbstractNeuron const&)> weightFunction = default_weights,
																		   Type eta = 0.15f, Type alpha = 0.5f);
}