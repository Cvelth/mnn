#pragma once
#include "AbstractTest.hpp"

#include "Automatization.hpp"
#include "AbstractNeuron.hpp"
#include "AbstractLayerNetwork.hpp"
#include "Exceptions.hpp"

namespace MNNT {
	/*
	 * Neural Network test realization allowing to check the stability of the Network.
	 * During learning it feeds the network with the same list of static input data which is unchangable during object existence
	 * The learning process is executed of the output data array.
	 * Both inputs and outputs are passed in the Constructor.
	 *
	 * All the methods inherit base class's, For more details see AbstractTest.hpps
	 */
	class StaticDataTest : public AbstractTest {
	protected:
		MNN::AbstractLayerNetwork* m_network;
		std::initializer_list<float> m_inputs;
		std::initializer_list<float> m_outputs;
	public:
		/*
		 * Constructor of static test class.
		 * Parameters:
		 * * static_inputs - the data passed into Network in every iteration.
		 * * static_output - expected output data for the input.
		 */
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
		virtual const size_t getOutputsNumber() override {
			return m_network->getOutputsNumber();
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