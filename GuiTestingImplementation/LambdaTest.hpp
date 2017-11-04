#pragma once
#include <functional>
#include <array>
#include "AbstractTest.hpp"
#include "AbstractLayerNetwork.hpp"
namespace mnnt {
	GenerateNewException(NoNetworkInsertedException)
	template<size_t Inputs, size_t Outputs, typename Test_Type = Type>
	class LambdaTest : public AbstractTest {
		mnn::AbstractLayerNetwork* m_network;
		std::array<Test_Type, Inputs> m_current_inputs;
		std::array<Test_Type, Outputs> m_current_outputs;
		std::function<void(std::array<Test_Type, Inputs> const&, std::array<Test_Type, Outputs>&)> m_function;
	protected:
		void newIteration() {
			for (auto it : m_current_inputs)
				it = m_random();
			m_function(m_current_inputs, m_current_outputs);
		}
	public:
		LambdaTest(std::function<void(std::array<Test_Type, Inputs> const&, std::array<Test_Type, Outputs>&)> function,
			size_t min_value = -1.f, size_t max_value = +1.f) : AbstractTest(), m_function(function), m_network(nullptr)
		{
			m_random.changeDistribution(min_value, max_value);
		}
		~LambdaTest() { if (m_network) delete m_network; }
		virtual void insertNeuralNetwork(mnn::AbstractLayerNetwork *network) override {
			m_network = network;
		}

		void calculate() {
			newIteration();
			if (m_network) m_network->calculateWithInputs(m_current_inputs);
			else throw Exceptions::NoNetworkInsertedException();
		}
		void learningProcess() {
			if (m_network) m_network->learningProcess(m_current_outputs);
			else throw Exceptions::NoNetworkInsertedException();
			calculate();
		}

		virtual size_t getInputsNumber() const override { return m_network->getOutputsNumber(); }
		virtual size_t getOutputsNumber() const override { return m_network->getOutputsNumber(); }
		virtual NeuronContainer<Type> getInputs() const override { return m_network->getInputs(); }
		virtual NeuronContainer<Type> getOutputs() const override { return m_network->getOutputs(); }
		virtual const float getOutput(size_t index) const override { return m_network->getOutput(index); }
		virtual const float getInput(size_t index) const override { return m_network->getInput(index); }
	};
}