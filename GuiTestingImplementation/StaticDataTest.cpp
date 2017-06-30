#include "StaticDataTest.hpp"

#include "AbstractLayerNetwork.hpp"
#include "Automatization.hpp"
#include "Exceptions.hpp"

MNNT::StaticDataTest::~StaticDataTest()
{
	delete m_network;
}

void MNNT::StaticDataTest::generateNeuralNetwork() {
	m_network = MNN::generateTypicalLayerNeuralNetwork(m_inputs.size(), m_outputs.size(), 0, 0, MNN::ConnectionPattern::EachFromPreviousLayerWithBias,
		[&](MNN::AbstractNeuron* n, MNN::AbstractNeuron* in) -> float {
		return m_random();
	}, 0.15f, 0.5f);
}

void MNNT::StaticDataTest::calculate() {
	m_network->calculateWithInputs(m_inputs);
}

void MNNT::StaticDataTest::learningProcess() {
	m_network->learningProcess(m_outputs);
	calculate();
}

const size_t MNNT::StaticDataTest::getOutputsNumber() {
	return m_network->getOutputsNumber();
}

const float * MNNT::StaticDataTest::getOutputs() {
	return m_network->getOutputs();
}

const float MNNT::StaticDataTest::getOutput(size_t index) {
	if (index < m_network->getOutputsNumber())
		return m_network->getOutputs()[index];
	else
		throw MNN::Exceptions::NonExistingIndexException();
}