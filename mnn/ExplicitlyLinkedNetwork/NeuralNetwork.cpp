#include "NeuralNetwork.hpp"
#include "mnn/exceptions.hpp"
DefineNewMNNException(UnimplementedFeature);

#include "neuron.hpp"
mnn::ExplicitlyLinkedNeuralNetwork::ExplicitlyLinkedNeuralNetwork(size_t input_number, size_t output_number)
	: NeuralNetworkInterface(input_number, output_number) {
	for (size_t i = 0; i < input_number; i++)
		m_input_neurons.push_back(std::make_shared<Neuron>());
	for (size_t i = 0; i < output_number; i++)
		m_output_neurons.push_back(std::make_shared<Neuron>());
}
mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::ExplicitlyLinkedBackpropagationNeuralNetwork(size_t input_number, size_t output_number)
	: BackpropagationNeuralNetworkInterface(input_number, output_number) {
	for (size_t i = 0; i < input_number; i++)
		m_input_neurons.push_back(std::make_shared<BackpropagationNeuron>());
	for (size_t i = 0; i < output_number; i++)
		m_output_neurons.push_back(std::make_shared<BackpropagationNeuron>());
}

void mnn::ExplicitlyLinkedNeuralNetwork::process() {
	auto it1 = m_inputs.cbegin();
	auto it2 = m_input_neurons.begin();
	while (it1 != m_inputs.cend() || it2 != m_input_neurons.end()) {
		**it2 = *it1;
		it1++; it2++;
	}

	auto it3 = m_outputs.begin();
	auto it4 = m_output_neurons.begin();
	while (it3 != m_outputs.end() || it4 != m_output_neurons.end()) {
		*it3 = (*it4)->value(true);
		it3++; it4++;
	}
}
void mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::process() {
	auto it1 = m_inputs.cbegin();
	auto it2 = m_input_neurons.begin();
	while (it1 != m_inputs.cend() || it2 != m_input_neurons.end()) {
		**it2 = *it1;
		it1++; it2++;
	}

	auto it3 = m_outputs.begin();
	auto it4 = m_output_neurons.begin();
	while (it3 != m_outputs.end() || it4 != m_output_neurons.end()) {
		*it3 = (*it4)->value(true);
		it3++; it4++;
	}
}

void mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::backpropagate(NeuronContainer<Value> const & _outputs) {
	//TO DO.
}