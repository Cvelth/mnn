#include "Neuron.hpp"
void mnn::Neuron::calculate(bool full) {
	Value temp = 0;
	for (auto &t : m_links)
		temp += t.unit->value(full) * t.weight;
	value(temp);
}

void mnn::BackpropagationNeuron::calculate(bool full) {
	Value temp = 0;
	for (auto &t : m_links)
		temp += t.unit->value(full) * t.weight;
	value(temp);
}

#include <algorithm>
mnn::Value mnn::BackpropagationNeuron::getWeightTo(BackpropagationNeuronInterface *neuron) {
	if (auto it = std::find_if(m_links.cbegin(), m_links.cend(), [neuron](auto const& l) {
		return *l.unit == *neuron;
	}); it != m_links.cend())
		return it->weight;
	else
		return 0.0;
}
void mnn::BackpropagationNeuron::recalculateWeights() {
	for (auto &l : m_links) {
		l.delta = m_eta * l.unit->value() * m_gradient
			+ m_alpha * l.delta;
		l();
	}
	changed();
}
void mnn::BackpropagationNeuron::calculateGradient(Value const& expectedValue) {
	m_gradient = (expectedValue - value()) * normalization_derivative(value());
}
void mnn::BackpropagationNeuron::calculateGradient(std::function<Value(std::function<Value(BackpropagationNeuronInterface&)>)> gradient_sum) {
	m_gradient = gradient_sum([this](auto &n) -> Value {
		return n.getWeightTo(this) * n.gradient();
	}) * normalization_derivative(value());
}

#include "mnn/storage/Storage.hpp"
std::ostream& mnn::Neuron::to_stream(std::ostream &output) const {
	output << " " << short(typecodes::neuron) << ' ' << id() << '\n';
	for (auto &l : m_links)
		output << "  " << short(typecodes::link) 
			<< ' ' << l.unit->id() << ' ' << l.weight << '\n';
	return output;
}
std::ostream& mnn::BackpropagationNeuron::to_stream(std::ostream &output) const {
	output << " " << short(typecodes::neuron_backpropagation) << ' ' << id() << '\n';
	for (auto &l : m_links)
		output << "  " << short(typecodes::link_backpropagation) 
			<< ' ' << l.unit->id() << ' ' << l.weight << ' ' << l.delta << '\n';
	return output;
}