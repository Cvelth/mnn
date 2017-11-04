#pragma once
#include "Shared.hpp"
#include "RandomEngine.hpp"
namespace mnn { class AbstractLayerNetwork; }
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

		//Changes the neural network being used by the test.
		//Pass *nullptr* to remove any network installed.
		virtual void insertNeuralNetwork(mnn::AbstractLayerNetwork *network) abstract;

		//One iteration of network using the *static_input* data.
		virtual void calculate() abstract;

		//One iteration of learning using the *static_output* data.
		virtual void learningProcess() abstract;

		//*number_of_iterations* of *learningProcess* fuction calls.
		inline void repeatedLearning(size_t number_of_iterations) {
			for (size_t i = 0; i < number_of_iterations; i++)
				learningProcess();
		}

		virtual NeuronContainer<Type> getInputs() const = 0;
		virtual NeuronContainer<Type> getOutputs() const = 0;
		virtual size_t getInputsNumber() const = 0;
		virtual size_t getOutputsNumber() const = 0;
		virtual const float getInput(size_t index) const = 0;
		virtual const float getOutput(size_t index) const = 0;
	protected:
		RealRandomEngine m_random;
	};
}