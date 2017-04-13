#include "LayerNetwork.hpp"
#include "AbstractNeuron.hpp"
#include "Exceptions.hpp"
#include <initializer_list>

void MNN::LayerNetwork::newInputs(const std::initializer_list<float>& inputs, bool normalize) {
	if (inputs.size() != m_inputs->size())
		throw MNN::Exceptions::WrongInputsNumberException();

	auto counter = inputs.begin();
	m_inputs->for_each([&counter, &inputs, normalize](MNN::AbstractNeuron* n) {
		if (normalize)
			n->setValue(*(counter++));
		else
			n->setValueUnnormalized(*(counter++));
	});
}

void MNN::LayerNetwork::newInputs(size_t number, float * inputs, bool normalize) {
	if (number != m_inputs->size())
		throw MNN::Exceptions::WrongInputsNumberException();

	size_t counter = 0;
	m_inputs->for_each([&counter, &inputs, normalize](MNN::AbstractNeuron* n) {
		if (normalize)
			n->setValue(inputs[counter++]);
		else
			n->setValueUnnormalized(inputs[counter++]);
	});
}

void MNN::LayerNetwork::newInputs(const NetworkDataContainer<float>& inputs, bool normalize) {
	if (inputs.size() != m_inputs->size())
		throw MNN::Exceptions::WrongInputsNumberException();

	size_t counter = 0;
	m_inputs->for_each([&counter, &inputs, normalize](MNN::AbstractNeuron* n) {
		if (normalize)
			n->setValue(inputs[counter++]);
		else
			n->setValueUnnormalized(inputs[counter++]);
	});
}

void MNN::LayerNetwork::calculateWithInputs(const NetworkDataContainer<float>& inputs, bool normalize) {
	newInputs(inputs, normalize);
	calculate();
}

const float* MNN::LayerNetwork::getOutputs() const {
	float *res = new float[m_outputs->size()];
	size_t i = 0;
	m_outputs->for_each([&res, &i](MNN::AbstractNeuron* n) {
		res[i++] = n->value();
	});
	return res;
}
