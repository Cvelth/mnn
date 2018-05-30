#pragma once
#include "Shared.hpp"
#include "AbstractLayerNetwork.hpp"
#include "AbstractNetworkStorage.hpp"
#include "AbstractNeuron.hpp"
#include "SharedNetworkFunctionality.hpp"
namespace mnn {
	class LayerNetwork : public virtual AbstractLayerNetwork, public virtual SharedLayerNetworkFunctionality<AbstractNeuron> {
	public:
		using SharedLayerNetworkFunctionality::SharedLayerNetworkFunctionality;
		virtual std::string print() const override;
	};

	class BackpropagationLayerNetwork : public virtual AbstractBackpropagationLayerNetwork, public virtual SharedLayerNetworkFunctionality<AbstractBackpropagationNeuron> {
	public:
		using SharedLayerNetworkFunctionality::SharedLayerNetworkFunctionality;
		virtual std::string print() const override;

		void calculateGradients(const NeuronContainer<Type>& outputs) override {
			if (outputs.size() != m_storage.outputs()->size())
				throw Exceptions::IncorrectDataAmountException();

			auto it = outputs.begin();
			SharedLayerNetworkFunctionality<AbstractBackpropagationNeuron>::for_each_output([&it](AbstractBackpropagationNeuron& n) { n.calculateGradient(*(it++)); });

			AbstractLayer<AbstractBackpropagationNeuron>* nextLayer = m_storage.outputs();
			for_each_hidden([&nextLayer](AbstractLayer<AbstractBackpropagationNeuron>& l) {
				l.for_each([&nextLayer](AbstractBackpropagationNeuron& n) {
					n.calculateGradient([&nextLayer](std::function<Type(AbstractBackpropagationNeuron&)> calculate_unit) -> Type {
						Type sum = Type(0.f);
						nextLayer->for_each([&sum, &calculate_unit](AbstractBackpropagationNeuron& nn) {
							sum += calculate_unit(nn);
						});
						return sum;
					});
				});
				nextLayer = &l;
			}, false);
		}
		virtual void updateWeights() override {
			for_each_neuron([](AbstractBackpropagationNeuron& n) {
				n.recalculateWeights();
			}, false);
		}
	};
}