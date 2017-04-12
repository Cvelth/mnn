#include "Neuron.hpp"

void MNN::AbstractDataContainerNeuron::calculate() {
	float value = 0.f;
	for (Link t : m_links)
		value += t.unit->value() * t.weight;
	this->setValue(value);
}
float MNN::AbstractDataContainerNeuron::normalize(const float & value) {
	return value; //Does Nothing
}
