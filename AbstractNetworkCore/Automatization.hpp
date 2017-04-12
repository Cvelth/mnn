#pragma once
#include <functional>

namespace MNN {
	class AbstractNeuron;
	class AbstractLayerNetwork;
	enum class ConnectionPattern {
		NoDefaultConnection, EachFromPreviousLayerWithBias, EachFromPreviousLayerWithoutBias,
	};

	/*
	* The function generates and returns a pointer to a NeuralNetwork with *input_number* inputs, *output_number* outputs, *hidden_layers_number* hidden layers with *neurons_per_hidden_layer* neurons in each,
	* The default connection type is determined by *connection*.
	*
	* WeightFunction is an lambda-function-argument which takes the arguments:
	* * The neuron is being linked to
	* * The neuron is being linking
	* It returns a float -> weight of the Link.
	*
	* Default Links weight is equal to 1.f.
	*/
	AbstractLayerNetwork* generateTypicalLayerNeuralNetwork(size_t inputs_number, size_t outputs_number,
															size_t hidden_layers_number, size_t neurons_per_hidden_layer,
															ConnectionPattern connection = ConnectionPattern::EachFromPreviousLayerWithBias,
															std::function<float(AbstractNeuron*, AbstractNeuron*)> weightFunction = 
																[](AbstractNeuron* neuron, AbstractNeuron* input) -> float {
																	return 1.f;
																});
}