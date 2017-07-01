#include "LogicalFunctionTest.hpp"

#include "AbstractLayerNetwork.hpp"
#include "Automatization.hpp"
#include "Exceptions.hpp"


MNNT::LogicalFunctionTest::LogicalFunctionTest(LogicalFunction function)
	: AbstractTest(), m_function(function) {
	m_g = new std::mt19937_64(std::random_device()());
	m_d = new std::uniform_int_distribution<size_t>(false, true);
}

MNNT::LogicalFunctionTest::~LogicalFunctionTest()
{
	delete m_network;
	delete m_g;
	delete m_d;
}

void MNNT::LogicalFunctionTest::generateNeuralNetwork() {
	generateNeuralNetwork(2, 1, 0, 0);
}

void MNNT::LogicalFunctionTest::generateNeuralNetwork(size_t inputs, size_t outputs, size_t hidden, size_t per_hidden) {
	m_network = MNN::generateTypicalLayerNeuralNetwork(inputs, outputs, hidden, per_hidden, MNN::ConnectionPattern::EachFromPreviousLayerWithBias,
		[&](MNN::AbstractNeuron* n, MNN::AbstractNeuron* in) -> float {
		return m_random();
	}, 0.15f, 0.5f);
}

void MNNT::LogicalFunctionTest::calculate() {
	newIteration();
	m_network->calculateWithInputs({(float) m_current_i1, (float) m_current_i2});
}

void MNNT::LogicalFunctionTest::learningProcess() {
	m_network->learningProcess({(float) m_current_o});
	calculate();
}

const size_t MNNT::LogicalFunctionTest::getOutputsNumber() const {
	return m_network->getOutputsNumber();
}

const float* MNNT::LogicalFunctionTest::getOutputs() const {
	return m_network->getOutputs();
}

const float MNNT::LogicalFunctionTest::getOutput(size_t index) const {
	if (index < m_network->getOutputsNumber())
		return m_network->getOutputs()[index];
	else
		throw MNN::Exceptions::NonExistingIndexException();
}

const float MNNT::LogicalFunctionTest::getInput(size_t index) const {
	switch (index) {
	case 0: return m_current_i1;
	case 1: return m_current_i2;
	default: throw MNN::Exceptions::NonExistingIndexException();
	}
}

bool MNNT::LogicalFunctionTest::operation(bool i1, bool i2) {
	switch (m_function) {
		case MNNT::LogicalFunction::And: return i1 & i2;
		case MNNT::LogicalFunction::Or: return i1 | i2;
		case MNNT::LogicalFunction::ExOr: return i1 ^ i2;
	}
}

void MNNT::LogicalFunctionTest::newIteration() {
	m_current_i1 = (*m_d)(*m_g);
	m_current_i2 = (*m_d)(*m_g);
	m_current_o = operation(m_current_i1, m_current_i2);
}
