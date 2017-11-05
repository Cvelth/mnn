#pragma once
#include "Shared.hpp"
namespace mnn {
	class AbstractNetwork {
	public:
		virtual ~AbstractNetwork() {};
		virtual void calculate() =0;
		virtual void setInputs(NeuronContainer<Type> const& inputs, bool normalize = true) =0;
		inline void calculate(NeuronContainer<Type> const& inputs, bool normalize = true) {
			if (inputs.size() != getInputs().size())
				throw Exceptions::IncorrectDataAmountException();
			setInputs(inputs, normalize);
			calculate();
		}
		inline void learningProcess(NeuronContainer<Type> const& outputs) {
			if (outputs.size() != getOutputs().size())
				throw Exceptions::IncorrectDataAmountException();
			calculateGradients(outputs);
			updateWeights();
		}
		virtual void calculateGradients(NeuronContainer<Type> const& outputs) =0;
		virtual void updateWeights() =0;

		virtual NeuronContainer<Type> getInputs() const =0;
		virtual NeuronContainer<Type> getOutputs() const =0;
		virtual size_t getInputsNumber() const =0;
		virtual size_t getOutputsNumber() const =0;
		virtual const float getInput(size_t index) const = 0;
		virtual const float getOutput(size_t index) const = 0;
	};
}