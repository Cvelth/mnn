#pragma once
#include "AbstractTest.hpp"
#include <random>

namespace MNN {
	class AbstractLayerNetwork;
}

namespace mnnt {

	//Enum holding all the possible function values for the test.
	enum class LogicalFunction {
		And, Or, ExOr
	};

	/*
	Neural Network test realization allowing to check the adaptivity of the Network.
	During learning it feeds the network with tho binary numbers 
	The learning process is executed of the resulting function, choosen from the enum.
	Function is choosen in the Constructor
	
	All the methods inherit base class's, For more details see AbstractTest.hpps
	*/
	class LogicalFunctionTest : public AbstractTest {
	protected:
		MNN::AbstractLayerNetwork* m_network;
		LogicalFunction m_function;

		std::mt19937_64 *m_g;
		std::uniform_int_distribution<size_t> *m_d;

		bool m_current_i1, m_current_i2;
		bool m_current_o;

	protected:
		bool operation(bool i1, bool i2);
		void newIteration();

		virtual void generateNeuralNetwork(size_t inputs, size_t outputs, size_t hidden, size_t per_hidden) override;
	public:
		/*
		Constructor of logical function test class.
		Parameters:
		* function - choosen test function.
		*/
		LogicalFunctionTest(LogicalFunction function);
		~LogicalFunctionTest();
		virtual void generateNeuralNetwork();
		virtual void calculate() override;
		virtual void learningProcess() override;
		virtual const size_t getOutputsNumber() const override;
		virtual const float* getOutputs() const override;
		virtual const float getOutput(size_t index) const override;
		virtual const float getInput(size_t index) const;
	};
}