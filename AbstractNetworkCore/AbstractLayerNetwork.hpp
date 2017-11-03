#pragma once
#include <functional>
#include "AbstractNetwork.hpp"
namespace mnn {
	class AbstractNeuron;
	class AbstractLayer;
	class AbstractLayerNetwork : public AbstractNetwork {
	public:
		virtual ~AbstractLayerNetwork() {};
		inline virtual void addLayer(AbstractLayer* l) abstract;
		virtual void calculate() abstract;

		inline virtual void for_each_hidden(std::function<void(AbstractLayer&)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_hidden(std::function<void(AbstractLayer const&)> lambda, bool firstToLast = true) const abstract;
		inline virtual void for_each_layer(std::function<void(AbstractLayer&)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_layer(std::function<void(AbstractLayer const&)> lambda, bool firstToLast = true) const abstract;
		inline virtual void for_each_input(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_input(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const abstract;
		inline virtual void for_each_output(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_output(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const abstract;
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const abstract;
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const abstract;
	};
}