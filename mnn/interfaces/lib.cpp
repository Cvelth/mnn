#include "ExplicitlyLinkedNeuronInterface.hpp"
size_t mnn::NeuronInterface::NUMBER_OF_NEURONS_CREATED = 0;

#include <cmath>
mnn::Value mnn::NeuronInterface::normalize(Value const& value) {
	return tanh(value);
}