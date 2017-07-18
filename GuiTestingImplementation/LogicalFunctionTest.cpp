#include "LogicalFunctionTest.hpp"

#include "AbstractLayerNetwork.hpp"
#include "Automatization.hpp"
#include "Exceptions.hpp"


mnnt::LogicalFunctionTest::LogicalFunctionTest(LogicalFunction function)
	: AbstractTest(), m_function(function) {
	m_g = new std::mt19937_64(std::random_device()());
	m_d = new std::uniform_int_distribution<size_t>(false, true);
}

mnnt::LogicalFunctionTest::~LogicalFunctionTest()
{
	delete m_network;
	delete m_g;
	delete m_d;
}

void mnnt::LogicalFunctionTest::generateNeuralNetwork() {
	if (m_function == LogicalFunction::ExOr)
		generateNeuralNetwork(2, 1, 1, 2);
	else
		generateNeuralNetwork(2, 1, 0, 0);
}

void mnnt::LogicalFunctionTest::generateNeuralNetwork(size_t inputs, size_t outputs, size_t hidden, size_t per_hidden) {
	m_network = mnn::generateTypicalLayerNeuralNetwork(inputs, outputs, hidden, per_hidden, mnn::ConnectionPattern::EachFromPreviousLayerWithBias,
		[&](mnn::AbstractNeuron* n, mnn::AbstractNeuron* in) -> float {
		return m_random();
	}, 0.15f, 0.5f);
}

void mnnt::LogicalFunctionTest::calculate() {
	newIteration();
	m_network->calculateWithInputs({(float) m_current_i1, (float) m_current_i2});
}

void mnnt::LogicalFunctionTest::learningProcess() {
	m_network->learningProcess({(float) m_current_o});
	calculate();
}

const size_t mnnt::LogicalFunctionTest::getOutputsNumber() const {
	return m_network->getOutputsNumber();
}

const float* mnnt::LogicalFunctionTest::getOutputs() const {
	return m_network->getOutputs();
}

const float mnnt::LogicalFunctionTest::getOutput(size_t index) const {
	if (index < m_network->getOutputsNumber())
		return m_network->getOutputs()[index];
	else
		throw mnn::Exceptions::NonExistingIndexException();
}

const float mnnt::LogicalFunctionTest::getInput(size_t index) const {
	switch (index) {
	case 0: return m_current_i1;
	case 1: return m_current_i2;
	default: throw mnn::Exceptions::NonExistingIndexException();
	}
}

bool mnnt::LogicalFunctionTest::operation(bool i1, bool i2) {
	switch (m_function) {
		case mnnt::LogicalFunction::And: return i1 & i2;
		case mnnt::LogicalFunction::Or: return i1 | i2;
		case mnnt::LogicalFunction::ExOr: return i1 ^ i2;
		default: return false;
	}
}

void mnnt::LogicalFunctionTest::newIteration() {
	m_current_i1 = (*m_d)(*m_g);
	m_current_i2 = (*m_d)(*m_g);
	m_current_o = operation(m_current_i1, m_current_i2);
}
