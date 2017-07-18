#include "AbstractDataContainerNeuron.hpp"
#include "Sigmoids.hpp"
#include "AbstractLayer.hpp"

void mnn::AbstractDataContainerNeuron::calculate() {
	float value = 0.f;
	for (Link t : m_links)
		value += t.unit->value() * t.weight;
	this->setValue(value);
}
float mnn::AbstractDataContainerNeuron::normalize(const float & value) {
	return mnn::tanh_sigmoid(value);
}

mnn::AbstractDataContainerNeuron::~AbstractDataContainerNeuron() {
	
}

float mnn::AbstractDataContainerNeuron::getWeightTo(AbstractNeuron * neuron) {
	for (auto l : m_links)
		if (l.unit == neuron)
			return l.weight;
	return 0.f;
}

void mnn::AbstractDataContainerNeuron::recalculateWeights() {
	for_each([this](Link& l) {
		l.delta = m_constants.eta * l.unit->value() * m_gradient
				+ m_constants.alpha * l.delta;
		l.step();
	});

	changed();
}

void mnn::AbstractDataContainerNeuron::calculateGradient(float v) {
	m_gradient = (v - value()) * mnn::tanh_sigmoid_derivative(value());
}

void mnn::AbstractDataContainerNeuron::calculateGradient(AbstractLayer* nextLayer) {
	float sum = 0;
	nextLayer->for_each([&sum, this](AbstractNeuron* n) {
		sum += n->getWeightTo(this) * n->gradient();
	});

	m_gradient = sum * mnn::tanh_sigmoid_derivative(value());
}
