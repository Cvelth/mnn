#include "LayerNetwork.hpp"
#include "AbstractNeuron.hpp"
#include "Exceptions.hpp"
#include <initializer_list>
#include "ErrorSystem.h"

mnn::LayerNetwork::~LayerNetwork() {
	for (auto layer : m_layers)
		delete layer;
	if (m_inputs) delete m_inputs;
	if (m_outputs) delete m_outputs;
	if (m_errorSystem) delete m_errorSystem;
}

void mnn::LayerNetwork::newInputs(const std::initializer_list<float>& inputs, bool normalize) {
	if (inputs.size() != m_inputs->size())
		throw mnn::Exceptions::WrongInputsNumberException();

	auto counter = inputs.begin();
	m_inputs->for_each([&counter, &inputs, normalize](mnn::AbstractNeuron* n) {
		if (normalize)
			n->setValue(*(counter++));
		else
			n->setValueUnnormalized(*(counter++));
	});
}

void mnn::LayerNetwork::newInputs(size_t number, float * inputs, bool normalize) {
	if (number != m_inputs->size())
		throw mnn::Exceptions::WrongInputsNumberException();

	size_t counter = 0;
	m_inputs->for_each([&counter, &inputs, normalize](mnn::AbstractNeuron* n) {
		if (normalize)
			n->setValue(inputs[counter++]);
		else
			n->setValueUnnormalized(inputs[counter++]);
	});
}

void mnn::LayerNetwork::newInputs(const NetworkDataContainer<float>& inputs, bool normalize) {
	if (inputs.size() != m_inputs->size())
		throw mnn::Exceptions::WrongInputsNumberException();

	size_t counter = 0;
	m_inputs->for_each([&counter, &inputs, normalize](mnn::AbstractNeuron* n) {
		if (normalize)
			n->setValue(inputs[counter++]);
		else
			n->setValueUnnormalized(inputs[counter++]);
	});
}

void mnn::LayerNetwork::calculateWithInputs(const NetworkDataContainer<float>& inputs, bool normalize) {
	newInputs(inputs, normalize);
	calculate();
}

void mnn::LayerNetwork::learningProcess(const NetworkDataContainer<float>& outputs) {
	if (outputs.size() != getOutputsNumber())
		throw Exceptions::WrongOutputNumberException();

	float tempNetworkError = m_errorSystem->calculateNetworkError(this, outputs);
	calculateGradients(outputs);
	updateWeights();
}

void mnn::LayerNetwork::calculateGradients(const NetworkDataContainer<float>& outputs) {
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

void mnn::LayerNetwork::updateWeights() {
	for_each_neuron([](AbstractNeuron* n) {
		n->recalculateWeights();
	}, false);
}

void mnn::LayerNetwork::calculateGradients(const std::initializer_list<float>& outputs) {
	calculateGradients(NetworkDataContainer<float>(outputs));
}

float mnn::LayerNetwork::calculateNetworkError(const std::initializer_list<float>& outputs) {
	return m_errorSystem->calculateNetworkError(this, outputs);
}

const float* mnn::LayerNetwork::getOutputs() const {
	float *res = new float[m_outputs->size()];
	size_t i = 0;
	m_outputs->for_each([&res, &i](mnn::AbstractNeuron* n) {
		res[i++] = n->value();
	});
	return res;
}
