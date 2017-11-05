#include "LayerNetwork.hpp"
#include "AbstractNeuron.hpp"
mnn::LayerNetwork::~LayerNetwork() {
	for (auto layer : m_hidden) delete layer;
	if (m_inputs) delete m_inputs;
	if (m_outputs) delete m_outputs;
}
void mnn::LayerNetwork::setInputs(NeuronContainer<Type> const& inputs, bool normalize) {
	if (inputs.size() != m_inputs->size())
		throw Exceptions::IncorrectDataAmountException();

	auto it = inputs.begin();
	if (normalize) m_inputs->for_each([&it](mnn::AbstractNeuron& n) { n.setValue(*(it++)); });
	else m_inputs->for_each([&it](mnn::AbstractNeuron& n) { n.setValueUnnormalized(*(it++)); });
}
void mnn::LayerNetwork::calculateGradients(NeuronContainer<Type> const& outputs){
	if (outputs.size() != m_outputs->size())
		throw Exceptions::IncorrectDataAmountException();

	auto it = outputs.begin();
	for_each_output([&it](AbstractNeuron& n) { n.calculateGradient(*(it++)); });

	AbstractLayer* nextLayer = m_outputs;
	for_each_hidden([&nextLayer](AbstractLayer& l) {
		l.for_each([&nextLayer](AbstractNeuron& n) {
			n.calculateGradient([&nextLayer] (std::function<Type(AbstractNeuron&)> calculate_unit) -> Type {
				Type sum = Type(0.f);
				nextLayer->for_each([&sum, &calculate_unit](AbstractNeuron& nn) {
					sum += calculate_unit(nn);
				});
				return sum;
			});
		});
		nextLayer = &l;
	}, false);
}
void mnn::LayerNetwork::updateWeights() {
	for_each_neuron([](AbstractNeuron& n) {
		n.recalculateWeights();
	}, false);
}
NeuronContainer<Type> mnn::LayerNetwork::getInputs() const {
	NeuronContainer<Type> res;
	m_inputs->for_each([&res](mnn::AbstractNeuron& n) { res.push_back(n.value()); });
	return res;
}
NeuronContainer<Type> mnn::LayerNetwork::getOutputs() const {
	NeuronContainer<Type> res;
	m_outputs->for_each([&res](mnn::AbstractNeuron& n) { res.push_back(n.value()); });
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
#include "TypeCodes.hpp"
#include <sstream>
std::string mnn::LayerNetwork::print() const {
	std::ostringstream res;
	res << LayerNetworkTypeCode << '\n';
	res << InputsTypeCode << " " << m_inputs->print();
	res << OutputsTypeCode << " " << m_outputs->print();
	res << HiddenTypeCode << " " << m_hidden.size() << '\n';
	for (auto it : m_hidden)
		res << it->print() << '\n';
	res << LayerNetworkTypeCode;
	return res.str();
}