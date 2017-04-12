#pragma once
#include "NetworkContainer.hpp"
#include "AbstractLayerNetwork.hpp"
#include "AbstractLayer.hpp"

namespace MNN {
	class LayerNetwork : public AbstractLayerNetwork {
	private:

	protected:
		NetworkDataContainer<AbstractLayer*> m_layers;
		AbstractLayer* m_inputs;
		AbstractLayer* m_outputs;
	public:
		explicit LayerNetwork(AbstractLayer* inputs, AbstractLayer* outputs) : m_inputs(inputs), m_outputs(outputs) {}
		inline virtual void addLayer(AbstractLayer* l) override {
			m_layers.push_back(l);
		}
		inline virtual void addLayers(const NetworkDataContainer<AbstractLayer*>& c) {
			for (auto t : c)
				this->addLayer(t);
		}
		explicit LayerNetwork(AbstractLayer* inputs, AbstractLayer* outputs, const NetworkDataContainer<AbstractLayer*>& c) : LayerNetwork(inputs, outputs) {
			this->addLayers(c);
		}

		virtual void calculate() override {
			m_outputs->calculate();
		}

		inline virtual void for_each_hidden(std::function<void(AbstractLayer*)> lambda) override {
			for (auto it : m_layers) {
				lambda(it);
			}
		}
		inline virtual void for_each(std::function<void(AbstractLayer*)> lambda) override {
			lambda(m_inputs);
			for (auto it : m_layers) {
				lambda(it);
			}
			lambda(m_outputs);
		}
		inline virtual void for_each_input(std::function<void(AbstractNeuron*)> lambda) override {
			m_inputs->for_each(lambda);
		}
		inline virtual void for_each_output(std::function<void(AbstractNeuron*)> lambda) override {
			m_outputs->for_each(lambda);
		}
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron*)> lambda) override {
			m_inputs->for_each(lambda);
			for (auto it : m_layers) {
				it->for_each(lambda);
			}
			m_outputs->for_each(lambda);
		}
	};
}