#include "Neuron.hpp"
#include "Sigmoids.hpp"
void mnn::Neuron::calculate() {
	float value = 0.f;
	for (Link t : m_links)
		value += t.unit->value() * t.weight;
	setValue(value);
}
Type mnn::Neuron::normalize(Type const& value) {
	return mnn::tanh_sigmoid(value);
}
Type mnn::Neuron::getWeightTo(AbstractNeuron *neuron) {
	for (auto l : m_links)
		if (l.unit == neuron)
			return l.weight;
	return 0.f;
}
void mnn::Neuron::recalculateWeights() {
	for_each_link([this](Link& l) {
		l.delta = m_eta * l.unit->value() * m_gradient
				+ m_alpha * l.delta;
		l.step();
	});
	changed();
}
void mnn::Neuron::calculateGradient(Type const& expectedValue) {
	m_gradient = (expectedValue - value()) * mnn::tanh_sigmoid_derivative(value());
}
//[[deprecated]]
#include "AbstractLayer.hpp"
void mnn::Neuron::calculateGradient(AbstractLayer* nextLayer) {
	float sum = 0;
	nextLayer->for_each([&sum, this](AbstractNeuron& n) {
		sum += n.getWeightTo(this) * n.gradient();
	});

	m_gradient = sum * mnn::tanh_sigmoid_derivative(value());
}