#include "Neuron.hpp"
#include "Sigmoids.hpp"
void mnn::Neuron::calculate() {
	float value = 0.f;
	for (Link& t : *m_links)
		value += t.unit->value() * t.weight;
	setValue(value);
}
Type mnn::Neuron::normalize(Type const& value) {
	return mnn::tanh_sigmoid(value);
}

void mnn::BackpropagationNeuron::calculate() {
	float value = 0.f;
	for (BackpropagationLink& t : *m_links)
		value += t.unit->value() * t.weight;
	setValue(value);
}
Type mnn::BackpropagationNeuron::normalize(Type const& value) {
	return mnn::tanh_sigmoid(value);
}
Type mnn::BackpropagationNeuron::getWeightTo(AbstractBackpropagationNeuron *neuron) {
	for (BackpropagationLink& l : *m_links)
		if (l.unit == neuron)
			return l.weight;
	return 0.f;
}
void mnn::BackpropagationNeuron::recalculateWeights() {
	for_each_link([this](BackpropagationLink& l) {
		l.delta = m_eta * l.unit->value() * m_gradient
			+ m_alpha * l.delta;
		l.step();
	});
	changed();
}
void mnn::BackpropagationNeuron::calculateGradient(Type const& expectedValue) {
	m_gradient = (expectedValue - value()) * mnn::tanh_sigmoid_derivative(value());
}
void mnn::BackpropagationNeuron::calculateGradient(std::function<Type(std::function<Type(AbstractBackpropagationNeuron&)>)> gradient_sum) {
	m_gradient = gradient_sum([this](AbstractBackpropagationNeuron &n) -> Type {
		return n.getWeightTo(this) * n.gradient();
	}) * mnn::tanh_sigmoid_derivative(value());
}
#include <sstream>
#include "TypeCodes.hpp"
std::string mnn::Neuron::print() const {
	std::ostringstream res;
	res << "\t\t" << NeuronTypeCode << ' ' << id() << ' ' << m_links->size() << '\n';
	for (auto& it : *m_links)
		res << "\t\t\t" << LinkTypeCode << ' ' << it.unit->id() << ' ' << it.weight << '\n';
	return res.str();
}
std::string mnn::BackpropagationNeuron::print() const {
	std::ostringstream res;
	res << "\t\t" << BackpropagationNeuronTypeCode << ' ' << id() << ' '
		<< m_eta << ' ' << m_alpha << ' ' << m_links->size() << '\n';
	for (auto& it : *m_links)
		res << "\t\t\t" << BackpropagationLinkTypeCode << ' ' << it.unit->id()
		<< ' ' << it.weight << ' ' << it.delta << '\n';
	return res.str();
}