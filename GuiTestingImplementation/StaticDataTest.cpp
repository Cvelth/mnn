#include "StaticDataTest.hpp"
#include <initializer_list>

#include "AbstractLayerNetwork.hpp"
#include "Automatization.hpp"
#include "Exceptions.hpp"

MNNT::AbstractStaticTest::~AbstractStaticTest()
{
	delete m_network;
}

void MNNT::AbstractStaticTest::generateNeuralNetwork(size_t inputs, size_t outputs, size_t hidden, size_t per_hidden) {
	m_network = MNN::generateTypicalLayerNeuralNetwork(inputs, outputs, hidden, per_hidden, MNN::ConnectionPattern::EachFromPreviousLayerWithBias,
		[&](MNN::AbstractNeuron* n, MNN::AbstractNeuron* in) -> float {
			return m_random();
	}, 0.15f, 0.5f);
}
const size_t MNNT::AbstractStaticTest::getOutputsNumber() const {
	return m_network->getOutputsNumber();
}

const float * MNNT::AbstractStaticTest::getOutputs() const {
	return m_network->getOutputs();
}

const float MNNT::AbstractStaticTest::getOutput(size_t index) const {
	if (index < m_network->getOutputsNumber())
		return m_network->getOutputs()[index];
	else
		throw MNN::Exceptions::NonExistingIndexException();
}

void MNNT::StaticDataTest::generateNeuralNetwork() {
	AbstractStaticTest::generateNeuralNetwork(m_inputs.size(), m_outputs.size(), 0, 0);
}

void MNNT::StaticDataTest::calculate() {
	m_network->calculateWithInputs(m_inputs);
}

void MNNT::StaticDataTest::learningProcess() {
	m_network->learningProcess(m_outputs);
	calculate();
}

void MNNT::StaticMultiDataTest::addDataSet(const std::initializer_list<float>& inputs, const std::initializer_list<float>& outputs) {
	if (inputs.size() == 0 || outputs.size() == 0)
		throw Exceptions::EmptyDataException();

	m_inputs.push_back(inputs);
	m_outputs.push_back(outputs);
}

void MNNT::StaticMultiDataTest::generateNeuralNetwork() {
	if (m_inputs.size() == 0 || m_outputs.size() == 0)
		throw Exceptions::NoDataException();
	if (!checkData())
		throw Exceptions::IncorrectDataSizeException();

	AbstractStaticTest::generateNeuralNetwork(m_inputs[0].size(), m_outputs[0].size(), 0, 0);
}

void MNNT::StaticMultiDataTest::incrementIndex() {
	if (m_current_index++ > m_inputs.size())
		m_current_index = 0;
}

bool MNNT::StaticMultiDataTest::checkData() const {
	size_t inputs_size = m_inputs[0].size();
	size_t outputs_size = m_outputs[0].size();

	for (int i = 1; i < m_inputs.size(); i++)
		if (m_inputs[i].size() != inputs_size || m_outputs[i].size() != outputs_size)
			return false;

	return true;
}

void MNNT::StaticMultiDataTest::calculate() {
	m_network->calculateWithInputs(m_inputs[m_current_index]);
	incrementIndex();
}

void MNNT::StaticMultiDataTest::learningProcess() {
	m_network->learningProcess(m_outputs[m_current_index - 1]);
	calculate();
}
