#pragma once
#include <functional>
#include "AbstractTest.hpp"
#include "AbstractLayerNetwork.hpp"
namespace mnnt {
	GenerateNewException(NoNetworkInsertedException)
	class LambdaTest : public AbstractTest {
		mnn::AbstractLayerNetwork* m_network;
		NeuronContainer<Type> m_current_inputs;
		NeuronContainer<Type> m_current_outputs;
		using FunctionType = std::function<void(NeuronContainer<Type> const&, NeuronContainer<Type>&)>;
		FunctionType m_function;
	protected:
		void newIteration() {
			for (auto it : m_current_inputs) it = m_random();
			m_function(m_current_inputs, m_current_outputs);
		}
		void check_network() const { if (!m_network) throw Exceptions::NoNetworkInsertedException(); }
	public:
		LambdaTest(FunctionType function, size_t min_value = -1.f, size_t max_value = +1.f) 
			: AbstractTest(), m_function(function), m_network(nullptr) {

			m_random.changeDistribution(min_value, max_value);
		}
		~LambdaTest() { if (m_network) delete m_network; }
		virtual void insertNeuralNetwork(mnn::AbstractLayerNetwork *network) override {
			m_network = network;
			if (m_network) {
				m_current_inputs.resize(getInputsNumber());
				m_current_outputs.resize(getOutputsNumber());
			}
		}

		void calculate() {
			newIteration();
			check_network();
			m_network->calculateWithInputs(m_current_inputs);
		}
		void learningProcess() {
			check_network();
			m_network->learningProcess(m_current_outputs);
			calculate();
		}

		virtual size_t getInputsNumber() const override { check_network(); return m_network->getOutputsNumber(); }
		virtual size_t getOutputsNumber() const override { check_network(); return m_network->getOutputsNumber(); }
		virtual NeuronContainer<Type> getInputs() const override { check_network(); return m_network->getInputs(); }
		virtual NeuronContainer<Type> getOutputs() const override { check_network(); return m_network->getOutputs(); }
		virtual const float getOutput(size_t index) const override { check_network(); return m_network->getOutput(index); }
		virtual const float getInput(size_t index) const override { check_network(); return m_network->getInput(index); }
	};
}