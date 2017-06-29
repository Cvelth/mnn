#pragma once
#include "RandomEngine.hpp"

namespace MNNT {

	/*
	 * Virtual Interface for all the Testing classes.
	 * Public methods are:
	 * * generateNeuralNetwork - for generation of the Basic Network using the *MNN::generateTypicalLayerNeuralNetwork* function (see Automatization.hpp for more detailed documentation).
	 * * calculate - to execute one calculation step on the network.
	 * * learningProcess - to execute one learning step on the network.
	 * * RepeatedLearning - to execute *learningProcess* *number_of_iteration* times.
	 * * getOutputs - to get an array with all the outoputs out of the network.
	 * * getOutput - to get one output data with *index* out of the network.
	 */
	class AbstractTest {
	public:
		AbstractTest() {}

		//Generates new neural network using *MNN::generateTypicalLayerNeuralNetwork* (see Automatization.hpp).
		virtual void generateNeuralNetwork() abstract;

		//One iteration of network using the *static_input* data.
		virtual void calculate() abstract;

		//One iteration of learning using the *static_output* data.
		virtual void learningProcess() abstract;

		//*number_of_iterations* of *learningProcess* fuction calls.
		virtual void repeatedLearning(size_t number_of_iterations) abstract;

		//Returns number of output neuron in the network. 
		virtual const size_t getOutputsNumber() abstract;

		//Returns array of outputs.
		virtual const float* getOutputs() abstract;

		//Returns one output with *index*.
		virtual const float getOutput(size_t index) abstract;
	protected:
		RealRandomEngine m_random;
	};
}