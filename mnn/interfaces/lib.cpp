#include "NeuronInterface.hpp"
size_t mnn::NeuronInterface::NUMBER_OF_NEURONS_CREATED = 0;

#include <cmath>
mnn::Value mnn::NeuronInterface::normalize(Value const& value) {
	return tanh(value);
}
mnn::Value mnn::NeuronInterface::normalization_derivative(Value const& value) {
	auto temp = tanh(value);
	return 1 - temp * temp;
}