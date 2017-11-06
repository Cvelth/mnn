#pragma once
#include <functional>
#include "AbstractLayer.hpp"
#include "AbstractNetwork.hpp"
namespace mnn {
	class AbstractNeuron;
	template <typename NeuronType>
	class AbstractNetworkLayerControlFunctions {
	public:
		virtual ~AbstractNetworkLayerControlFunctions() {}
		virtual void addHiddenLayer(AbstractLayer<NeuronType>* l) = 0;
		virtual void addHiddenLayers(LayerContainer<AbstractLayer<NeuronType>*> const& l) = 0;
		inline virtual AbstractLayer<NeuronType> const* getInputLayer() const = 0;
		inline virtual AbstractLayer<NeuronType> const* getOutputLayer() const = 0;

		inline virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const = 0;
		inline virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const = 0;
		inline virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const = 0;
		inline virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const = 0;
		inline virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const = 0;
		inline virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const = 0;
	};
	class AbstractNeuron;
	class AbstractLayerNetwork : public AbstractNetwork, public AbstractNetworkLayerControlFunctions<AbstractNeuron> {
		public: using AbstractNetwork::AbstractNetwork;
	};
	class AbstractBackpropagationNeuron;
	class AbstractBackpropagationLayerNetwork : public AbstractBackpropagationNetwork, public AbstractNetworkLayerControlFunctions<AbstractBackpropagationNeuron> {
		public: using AbstractBackpropagationNetwork::AbstractBackpropagationNetwork;
	};
}