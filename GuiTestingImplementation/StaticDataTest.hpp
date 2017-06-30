#pragma once
#include "AbstractTest.hpp"

namespace MNN {
	class AbstractLayerNetwork;
}

namespace MNNT {
	/*
	Neural Network test realization allowing to check the stability of the Network.
	During learning it feeds the network with the same list of static input data which is unchangable during object existence
	The learning process is executed of the output data array.
	Both inputs and outputs are passed in the Constructor.
	
	All the methods inherit base class's, For more details see AbstractTest.hpps
	*/
	class StaticDataTest : public AbstractTest {
	protected:
		MNN::AbstractLayerNetwork* m_network;
		std::initializer_list<float> m_inputs;
		std::initializer_list<float> m_outputs;
	public:
		/*
		Constructor of static test class.
		Parameters:
		* static_inputs - the data passed into Network in every iteration.
		* static_output - expected output data for the input.
		*/
		StaticDataTest(std::initializer_list<float> static_inputs, std::initializer_list<float> static_outputs) 
						: AbstractTest(), m_inputs(static_inputs), m_outputs(static_outputs) {}
		~StaticDataTest();
		virtual void generateNeuralNetwork() override;
		virtual void calculate() override;
		virtual void learningProcess() override;
		virtual const size_t getOutputsNumber() override;
		virtual const float* getOutputs() override;
		virtual const float getOutput(size_t index) override;
	};
}