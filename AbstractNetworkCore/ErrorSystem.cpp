#include "ErrorSystem.h"
#include "AbstractNeuron.hpp"
#include "AbstractLayerNetwork.hpp"
#include "Exceptions.hpp"

float mnn::ErrorSystems::AbstractErrorSystem::calculateNetworkError(AbstractLayerNetwork * network, const NetworkDataContainer<float>& outputs) {
	if (outputs.size() != network->getOutputsNumber())
		throw Exceptions::WrongOutputNumberException();
	return calculate(network, outputs);
}

float mnn::ErrorSystems::MeanSquareError::calculate(AbstractLayerNetwork * network, const NetworkDataContainer<float>& outputs) {
	float res = 0.f;
	unsigned int i = 0;
	network->for_each_output([&res, &outputs, &i](mnn::AbstractNeuron* n) {
		auto temp = outputs[i] - n->value();
		res += temp * temp;
	});
	return res / 2.f;
}


float mnn::ErrorSystems::RootMeanSquareError::calculate(AbstractLayerNetwork * network, const NetworkDataContainer<float>& outputs) {
	float res = 0.f;
	unsigned int i = 0;
	network->for_each_output([&res, &outputs, &i](mnn::AbstractNeuron* n) {
		auto temp = outputs[i] - n->value();
		res += temp * temp;
	});
	return sqrt(res / outputs.size());
}
