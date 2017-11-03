#pragma once
#include "Constants.hpp"
namespace mnn {
	class AbstractNetwork {
	public:
		virtual ~AbstractNetwork() {};
		virtual void calculate() abstract;
		virtual void newInputs(NetworkContainer<Type> const& inputs, bool normalize = true) abstract;
		inline void calculateWithInputs(NetworkContainer<Type> const& inputs, bool normalize = true) {
			newInputs(inputs, normalize);
			calculate();
		}
		inline void learningProcess(NetworkContainer<Type> const& outputs) {
			calculateGradients(outputs);
			updateWeights();
		}
		virtual void calculateGradients(NetworkContainer<Type> const& outputs) abstract;
		virtual void updateWeights() abstract;

		virtual NetworkContainer<Type> const getInputs() const abstract;
		virtual NetworkContainer<Type> const getOutputs() const abstract;
	};
}