#include "ErrorSystem.h"
#include "AbstractNeuron.hpp"
#include "AbstractLayerNetwork.hpp"
#include "Exceptions.hpp"

float MNN::MeanSquareError::calculate(AbstractLayerNetwork * network, const NetworkDataContainer<float>& outputs) {
	float res = 0.f;
	unsigned int i = 0;
	network->for_each_output([&res, &outputs, &i](MNN::AbstractNeuron* n) {
		auto temp = outputs[i] - n->value();
		res += temp * temp;
	});
	return res / 2.f;
}


float MNN::RootMeanSquareError::calculate(AbstractLayerNetwork * network, const NetworkDataContainer<float>& outputs) {
	float res = 0.f;
	unsigned int i = 0;
	network->for_each_output([&res, &outputs, &i](MNN::AbstractNeuron* n) {
		auto temp = outputs[i] - n->value();
		res += temp * temp;
	});
	return sqrt(res / outputs.size());
}