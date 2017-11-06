#include "LayerNetwork.hpp"
#include "AbstractNeuron.hpp"
void mnn::LayerNetwork::setInputs(NeuronContainer<Type> const& inputs, bool normalize) {
	if (inputs.size() != m_layers.inputs()->size())
		throw Exceptions::IncorrectDataAmountException();

	auto it = inputs.begin();
	if (normalize) m_layers.inputs()->for_each([&it](mnn::AbstractNeuron& n) { n.setValue(*(it++)); });
	else m_layers.inputs()->for_each([&it](mnn::AbstractNeuron& n) { n.setValueUnnormalized(*(it++)); });
}
NeuronContainer<Type> mnn::LayerNetwork::getInputs() const {
	NeuronContainer<Type> res;
	m_layers.inputs()->for_each([&res](mnn::AbstractNeuron& n) { res.push_back(n.value()); });
	return res;
}
NeuronContainer<Type> mnn::LayerNetwork::getOutputs() const {
	NeuronContainer<Type> res;
	m_layers.outputs()->for_each([&res](mnn::AbstractNeuron& n) { res.push_back(n.value()); });
	return res;
}
size_t mnn::LayerNetwork::getInputsNumber() const {
	return getInputLayer()->size();
}
size_t mnn::LayerNetwork::getOutputsNumber() const {
	return getOutputLayer()->size();
}
const float mnn::LayerNetwork::getInput(size_t index) const {
	return getInputLayer()->at(index);
}
const float mnn::LayerNetwork::getOutput(size_t index) const {
	return getOutputLayer()->at(index);
}

void mnn::BackpropagationLayerNetwork::setInputs(NeuronContainer<Type> const& inputs, bool normalize) {
	if (inputs.size() != m_layers.inputs()->size())
		throw Exceptions::IncorrectDataAmountException();

	auto it = inputs.begin();
	if (normalize) m_layers.inputs()->for_each([&it](mnn::AbstractNeuron& n) { n.setValue(*(it++)); });
	else m_layers.inputs()->for_each([&it](mnn::AbstractNeuron& n) { n.setValueUnnormalized(*(it++)); });
}
void mnn::BackpropagationLayerNetwork::calculateGradients(NeuronContainer<Type> const& outputs) {
	if (outputs.size() != m_layers.outputs()->size())
		throw Exceptions::IncorrectDataAmountException();

	auto it = outputs.begin();
	for_each_output([&it](AbstractBackpropagationNeuron& n) { n.calculateGradient(*(it++)); });

	AbstractLayer<AbstractBackpropagationNeuron>* nextLayer = m_layers.outputs();
	for_each_hidden([&nextLayer](AbstractLayer<AbstractBackpropagationNeuron>& l) {
		l.for_each([&nextLayer](AbstractBackpropagationNeuron& n) {
			n.calculateGradient([&nextLayer](std::function<Type(AbstractBackpropagationNeuron&)> calculate_unit) -> Type {
				Type sum = Type(0.f);
				nextLayer->for_each([&sum, &calculate_unit](AbstractBackpropagationNeuron& nn) {
					sum += calculate_unit(nn);
				});
				return sum;
			});
		});
		nextLayer = &l;
	}, false);
}
void mnn::BackpropagationLayerNetwork::updateWeights() {
	for_each_neuron([](AbstractBackpropagationNeuron& n) {
		n.recalculateWeights();
	}, false);
}
NeuronContainer<Type> mnn::BackpropagationLayerNetwork::getInputs() const {
	NeuronContainer<Type> res;
	m_layers.inputs()->for_each([&res](mnn::AbstractBackpropagationNeuron& n) { res.push_back(n.value()); });
	return res;
}
NeuronContainer<Type> mnn::BackpropagationLayerNetwork::getOutputs() const {
	NeuronContainer<Type> res;
	m_layers.outputs()->for_each([&res](mnn::AbstractBackpropagationNeuron& n) { res.push_back(n.value()); });
	return res;
}
size_t mnn::BackpropagationLayerNetwork::getInputsNumber() const {
	return getInputLayer()->size();
}
size_t mnn::BackpropagationLayerNetwork::getOutputsNumber() const {
	return getOutputLayer()->size();
}
const float mnn::BackpropagationLayerNetwork::getInput(size_t index) const {
	return getInputLayer()->at(index);
}
const float mnn::BackpropagationLayerNetwork::getOutput(size_t index) const {
	return getOutputLayer()->at(index);
}
#include "TypeCodes.hpp"
#include <sstream>
std::string mnn::LayerNetwork::print() const {
	std::ostringstream res;
	res << LayerNetworkTypeCode << '\n';
	res << InputsTypeCode << " " << m_layers.inputs()->print();
	res << OutputsTypeCode << " " << m_layers.outputs()->print();
	res << HiddenTypeCode << " " << m_layers->size() << '\n';
	for (auto& it : *m_layers)
		res << it->print() << '\n';
	res << LayerNetworkTypeCode;
	return res.str();
}
std::string mnn::BackpropagationLayerNetwork::print() const {
	std::ostringstream res;
	res << BackpropagationLayerNetworkTypeCode << '\n';
	res << InputsTypeCode << " " << m_layers.inputs()->print();
	res << OutputsTypeCode << " " << m_layers.outputs()->print();
	res << HiddenTypeCode << " " << m_layers->size() << '\n';
	for (auto& it : *m_layers)
		res << it->print() << '\n';
	res << BackpropagationLayerNetworkTypeCode;
	return res.str();
}