#include "ErrorSystem.h"
#include "AbstractNeuron.hpp"
#include "AbstractLayerNetwork.hpp"
#include "Exceptions.hpp"

float MNN::RootMeanSquareError::calculateNetworkError(const NetworkDataContainer<float>& outputs) {
	if (outputs.size() != m_network->getOutputsNumber())
		throw Exceptions::WrongOutputNumberException();
	float res = 0.f;
	unsigned int i = 0;
	m_network->for_each_output([&res, &outputs, &i](MNN::AbstractNeuron* n) {
		auto temp = outputs[i] - n->value();
		res += temp * temp;
	});
	return sqrt(res / outputs.size());
}
