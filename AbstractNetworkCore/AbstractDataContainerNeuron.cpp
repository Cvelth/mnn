#include "AbstractDataContainerNeuron.hpp"
#include "Sigmoids.hpp"
#include "AbstractLayer.hpp"

void MNN::AbstractDataContainerNeuron::calculate() {
	float value = 0.f;
	for (Link t : m_links)
		value += t.unit->value() * t.weight;
	this->setValue(value);
}
float MNN::AbstractDataContainerNeuron::normalize(const float & value) {
	return MNN::tanh_sigmoid(value);
}

MNN::AbstractDataContainerNeuron::~AbstractDataContainerNeuron() {
	
}

float MNN::AbstractDataContainerNeuron::getWeightTo(AbstractNeuron * neuron) {
	for (auto l : m_links)
		if (l.unit == neuron)
			return l.weight;
	return 0.f;
}

void MNN::AbstractDataContainerNeuron::recalculateWeights() {
	for_each([this](Link& l) {
		l.delta = m_constants.eta * l.unit->value() * m_gradient
				+ m_constants.alpha * l.delta;
		l.step();
	});

	changed();
}

void MNN::AbstractDataContainerNeuron::calculateGradient(float v) {
	m_gradient = (v - value()) * MNN::tanh_sigmoid_derivative(value());
}

void MNN::AbstractDataContainerNeuron::calculateGradient(AbstractLayer* nextLayer) {
	float sum = 0;
	nextLayer->for_each([&sum, this](AbstractNeuron* n) {
		sum += n->getWeightTo(this) * n->gradient();
	});

	m_gradient = sum * MNN::tanh_sigmoid_derivative(value());
}
