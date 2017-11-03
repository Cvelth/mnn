#pragma once
#include <initializer_list>
namespace mnn {
	//Abstract class for storing typical network data to be inherited by other classes. 
	//Handles inner calculation of the network parts for value generation and learning, neuron storring.
	class AbstractNetwork {
	public:
		virtual ~AbstractNetwork() {};

		//Tells all the neurons to recalculate their values, if they were changed.
		virtual void calculate() abstract;

		//Adds inputs to the network.
		virtual void newInputs(const std::initializer_list<float>& inputs, bool normalize = true) abstract;
		//Adds inputs to the network.
		virtual void newInputs(size_t number, float* inputs, bool normalize = true) abstract;
		//Calculates new outputs with the data given in *inputs*.
		void calculateWithInputs(const std::initializer_list<float>& inputs, bool normalize = true) {
			newInputs(inputs, normalize);
			calculate();
		}
		//Calculates new outputs with the data given in *inputs*.
		void calculateWithInputs(size_t number, float* inputs, bool normalize = true) {
			newInputs(number, inputs, normalize);
			calculate();
		}
		//Executes learning process for every neuron.
		void learningProcess(const std::initializer_list<float>& outputs) {
			float tempNetworkError = calculateNetworkError(outputs);
			calculateGradients(outputs);
			updateWeights();
		}
		//Returns network error.
		virtual float calculateNetworkError(const std::initializer_list<float>& outputs) abstract;
		//Runs gradient calculation for every neuron.
		virtual void calculateGradients(const std::initializer_list<float>& outputs) abstract;
		//Updates all the weights accordingly to the learnings errors.
		virtual void updateWeights() abstract;

		//Return the number of inputs in the network.
		virtual const size_t getInputsNumber() const abstract;
		//Return the number of outputs in the network.
		virtual const size_t getOutputsNumber() const abstract;

		//Returns outputs array.
		virtual const float* getOutputs() const abstract;
	};
}