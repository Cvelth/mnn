#pragma once
#include "RandomEngine.hpp"

namespace MNNT {
	class AbstractTest {
	public:
		AbstractTest() {}
		virtual void generateNeuralNetwork() abstract;
		virtual void calculate() abstract;
		virtual void learningProcess() abstract;
		virtual void repeatedLearning(size_t number_of_iterations) abstract;
		virtual const float* getOutputs() abstract;
		virtual const float getOutput(size_t index) abstract;
	protected:
		RealRandomEngine m_random;
	};
}