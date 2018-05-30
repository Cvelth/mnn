#pragma once
#include <functional>
#include "AbstractLayer.hpp"
namespace mnn {
	template <typename NeuronType>
	class AbstractMatrixNetworkControlFunctions {
	public:
		virtual ~AbstractMatrixNetworkControlFunctions() {}
		virtual AbstractLayer<NeuronType> const* getInputLayer() const = 0;
		virtual AbstractLayer<NeuronType> const* getOutputLayer() const = 0;

		virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool firstToLast = true) = 0;
		virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const = 0;
		virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) = 0;
		virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const = 0;
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) = 0;
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const = 0;
		virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) = 0;
		virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const = 0;
	};
	template <typename NeuronType>
	class AbstractLayerNetworkControlFunctions : public AbstractMatrixNetworkControlFunctions<NeuronType> {
	public:
		virtual ~AbstractLayerNetworkControlFunctions() {}
		virtual void addHiddenLayer(AbstractLayer<NeuronType>* l) = 0;
		virtual void addHiddenLayers(LayerContainer<AbstractLayer<NeuronType>*> const& l) = 0;

		virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) = 0;
		virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const = 0;
		virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) = 0;
		virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const = 0;
	};
}