#pragma once
#include <functional>
#include "AbstractNetwork.hpp"

namespace MNN {
	class AbstractNeuron;
	class AbstractLayer;

	class AbstractLayerNetwork : public AbstractNetwork {
	private:

	public:
		inline virtual void addLayer(AbstractLayer* l) abstract;
		virtual void calculate() abstract;

		inline virtual void for_each_hidden(std::function<void(AbstractLayer*)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_layer(std::function<void(AbstractLayer*)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_input(std::function<void(AbstractNeuron*)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_output(std::function<void(AbstractNeuron*)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron*)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron*)> lambda, bool firstToLast = true) abstract;
	};
}