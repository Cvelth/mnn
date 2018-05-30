#pragma once
#include "AbstractMatrixNetwork.hpp"
#include "SharedNetworkFunctionality.hpp"
namespace mnn {	
	class MatrixNetwork : public virtual AbstractMatrixNetwork, public virtual SharedMatrixNetworkFunctionality<AbstractNeuron> {
	public:
		using SharedMatrixNetworkFunctionality::SharedMatrixNetworkFunctionality;
		virtual std::string print() const override;
	};
	class BackpropagationMatrixNetwork : public virtual AbstractBackpropagationMatrixNetwork, public virtual SharedMatrixNetworkFunctionality<AbstractBackpropagationNeuron> {
	public:
		using SharedMatrixNetworkFunctionality::SharedMatrixNetworkFunctionality;
		virtual std::string print() const override;

		void calculateGradients(const NeuronContainer<Type>& outputs) override {
			//To be implemented.
		}
		virtual void updateWeights() override {
			//To be implemented.
		}
	};
}