#pragma once
#include "AbstractTest.hpp"

#include "Automatization.hpp"
#include "AbstractNeuron.hpp"
#include "AbstractLayerNetwork.hpp"
#include "Exceptions.hpp"

namespace MNNT {
	class StaticDataTest : public AbstractTest {
	protected:
		MNN::AbstractLayerNetwork* m_network;
		std::initializer_list<float> m_inputs;
		std::initializer_list<float> m_outputs;
	public:
		StaticDataTest(std::initializer_list<float> static_inputs, std::initializer_list<float> static_outputs) 
						: AbstractTest(), m_inputs(static_inputs), m_outputs(static_outputs) {}
		virtual void generateNeuralNetwork() override {
			m_network = MNN::generateTypicalLayerNeuralNetwork(m_inputs.size(), m_outputs.size(), 0, 0, MNN::ConnectionPattern::EachFromPreviousLayerWithBias,
				[&](MNN::AbstractNeuron* n, MNN::AbstractNeuron* in) -> float {
					return m_random();
				}, 0.15f, 0.5f);
		}
		virtual void calculate() override {
			m_network->calculateWithInputs(m_inputs);
		}
		virtual void learningProcess() override {
			m_network->learningProcess(m_outputs);
			calculate();
		}
		virtual void repeatedLearning(size_t number_of_iterations) override {
			for (size_t i = 0; i < number_of_iterations; i++)
				learningProcess();
		}
		virtual const float* getOutputs() override {
			return m_network->getOutputs();
		}
		virtual const float getOutput(size_t index) override {
			if (index < m_network->getOutputsNumber())
				return m_network->getOutputs()[index];
			else
				throw MNN::Exceptions::NonExistingIndexException();
		}
	};
}