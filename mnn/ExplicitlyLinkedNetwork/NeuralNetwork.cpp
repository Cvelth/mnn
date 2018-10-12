#include "NeuralNetwork.hpp"
void mnn::ExplicitlyLinkedNeuralNetwork::process() {
	for (auto &neuron : m_outputs)
		neuron->value(true);
}
