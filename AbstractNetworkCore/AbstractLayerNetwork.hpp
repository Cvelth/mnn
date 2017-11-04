#pragma once
#include <functional>
#include "AbstractNetwork.hpp"
namespace mnn {
	class AbstractNeuron;
	class AbstractLayer;
	class AbstractLayerNetwork : public AbstractNetwork {
	public:
		virtual ~AbstractLayerNetwork() {};
		virtual void addHiddenLayer(AbstractLayer* l) =0;
		virtual void addHiddenLayers(LayerContainer<AbstractLayer*> const& l) =0;
		virtual void calculate() =0;

		inline virtual AbstractLayer const* getInputLayer() const =0;
		inline virtual AbstractLayer const* getOutputLayer() const =0;

		inline virtual void for_each_hidden(std::function<void(AbstractLayer&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each_hidden(std::function<void(AbstractLayer&)> lambda, bool firstToLast = true) const =0;
		inline virtual void for_each_layer(std::function<void(AbstractLayer&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each_layer(std::function<void(AbstractLayer&)> lambda, bool firstToLast = true) const =0;
		inline virtual void for_each_input(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each_input(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const =0;
		inline virtual void for_each_output(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each_output(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const =0;
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const =0;
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const =0;
	};
}