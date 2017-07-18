#include "LayerNetwork.hpp"
#include "AbstractNeuron.hpp"
#include "Exceptions.hpp"
#include <initializer_list>
#include "ErrorSystem.h"

MNN::LayerNetwork::~LayerNetwork() {
	for (auto layer : m_layers)
		delete layer;
	if (m_inputs) delete m_inputs;
	if (m_outputs) delete m_outputs;
	if (m_errorSystem) delete m_errorSystem;
}

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

void MNN::LayerNetwork::learningProcess(const NetworkDataContainer<float>& outputs) {
	if (outputs.size() != getOutputsNumber())
		throw Exceptions::WrongOutputNumberException();

	float tempNetworkError = m_errorSystem->calculateNetworkError(this, outputs);
	calculateGradients(outputs);
	updateWeights();
}

void MNN::LayerNetwork::calculateGradients(const NetworkDataContainer<float>& outputs) {
	int i = 0;
	for_each_output([&outputs, &i](AbstractNeuron* n) {
		n->calculateGradient(outputs[i++]);
	});

	AbstractLayer* nextLayer = m_outputs;
	for_each_hidden([&nextLayer](AbstractLayer* l) {
		l->for_each([&nextLayer](AbstractNeuron* n) {
			n->calculateGradient(nextLayer);
		});
		nextLayer = l;
	}, false);
}

void MNN::LayerNetwork::updateWeights() {
	for_each_neuron([](AbstractNeuron* n) {
		n->recalculateWeights();
	}, false);
}

void MNN::LayerNetwork::calculateGradients(const std::initializer_list<float>& outputs) {
	calculateGradients(NetworkDataContainer<float>(outputs));
}

float MNN::LayerNetwork::calculateNetworkError(const std::initializer_list<float>& outputs) {
	return m_errorSystem->calculateNetworkError(this, outputs);
}

const float* MNN::LayerNetwork::getOutputs() const {
	float *res = new float[m_outputs->size()];
	size_t i = 0;
	m_outputs->for_each([&res, &i](MNN::AbstractNeuron* n) {
		res[i++] = n->value();
	});
	return res;
}
