#include "NeuralNetwork.hpp"
#include "mnn/exceptions.hpp"
DefineNewMNNException(UnimplementedFeature);

void mnn::ExplicitlyLinkedNeuralNetwork::process() {
	for (auto &neuron : m_outputs)
		neuron->value(true);
}

void mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::process() {
	for (auto &neuron : m_outputs)
		neuron->value(true);
}

void mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::calculateGradients(NeuronContainer<Value> const& _outputs) {
	//TO DO
	throw Exceptions::UnimplementedFeature();
}
void mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::calculateGradients(NeuronContainer<std::shared_ptr<NeuronInterface>> const& _outputs) {
	//TO DO
	throw Exceptions::UnimplementedFeature();
}
void mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::updateWeights() {
	for (auto &n : m_hidden)
		n->recalculateWeights();
	for (auto &n : m_outputs)
		std::dynamic_pointer_cast<BackpropagationNeuronInterface>(n)->recalculateWeights();
}
