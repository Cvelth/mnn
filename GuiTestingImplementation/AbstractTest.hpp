#pragma once
#include "RandomEngine.hpp"

namespace mnnt {

	/*
	 * Virtual Interface for all the Testing classes.
	 * Public methods are:
	 * * generateNeuralNetwork - for generation of the Basic Network using the *mnn::generateTypicalLayerNeuralNetwork* function (see Automatization.hpp for more detailed documentation).
	 * * calculate - to execute one calculation step on the network.
	 * * learningProcess - to execute one learning step on the network.
	 * * RepeatedLearning - to execute *learningProcess* *number_of_iteration* times.
	 * * getOutputs - to get an array with all the outoputs out of the network.
	 * * getOutput - to get one output data with *index* out of the network.
	 */
	class AbstractTest {
	public:
		AbstractTest() {}

		//Generates new neural network using *mnn::generateTypicalLayerNeuralNetwork* (see Automatization.hpp).
		virtual void generateNeuralNetwork(size_t inputs, size_t outputs, size_t hidden, size_t per_hidden) =0;

		//One iteration of network using the *static_input* data.
		virtual void calculate() =0;

		//One iteration of learning using the *static_output* data.
		virtual void learningProcess() =0;

		//*number_of_iterations* of *learningProcess* fuction calls.
		inline void repeatedLearning(size_t number_of_iterations) {
			for (size_t i = 0; i < number_of_iterations; i++)
				learningProcess();
		}

		//Returns number of output neuron in the network. 
		virtual const size_t getOutputsNumber() const =0;

		//Returns array of outputs.
		virtual const float* getOutputs() const =0;

		//Returns one output with *index*.
		virtual const float getOutput(size_t index) const =0;
	protected:
		RealRandomEngine m_random;
	};
}