#include "NeuralNetwork.hpp"
#include "layer.hpp"
void mnn::MatrixLayeredNeuralNetwork::process() {
	if (m_layers.back()->size() != m_outputs.size())
		throw Exceptions::MatrixStructureIsBroken("Size of the last layer must be equal to number of outputs.");
	
	auto current = m_inputs;
	for (auto &layer : m_layers)
		current = layer->process(current);
	
	auto it1 = m_outputs.begin();
	auto it2 = current.cbegin();
	while (it1 != m_outputs.end() || it2 != current.cend()) {
		*it1 = *it2;
		it1++; it2++;
	}
}

void mnn::MatrixLayeredNeuralNetwork::add_layer(size_t const& size, bool bias, Value const& minimum_weight_value, Value const& maximum_weight_value) {
	m_layers.push_back(std::make_shared<Layer>(size, m_layers.empty() ? m_inputs.size() : m_layers.size(), bias, minimum_weight_value, maximum_weight_value));
}