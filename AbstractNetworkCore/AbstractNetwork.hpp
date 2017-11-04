#pragma once
#include "Shared.hpp"
namespace mnn {
	class AbstractNetwork {
	public:
		virtual ~AbstractNetwork() {};
		virtual void calculate() abstract;
		virtual void setInputs(NeuronContainer<Type> const& inputs, bool normalize = true) abstract;
		inline void calculateWithInputs(NeuronContainer<Type> const& inputs, bool normalize = true) {
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
		virtual void calculateGradients(NeuronContainer<Type> const& outputs) abstract;
		virtual void updateWeights() abstract;

		virtual NeuronContainer<Type> getInputs() const abstract;
		virtual NeuronContainer<Type> getOutputs() const abstract;
	};
}