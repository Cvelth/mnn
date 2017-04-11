#pragma once
#include "NetworkContainer.hpp"
#include "AbstractLayerNetwork.hpp"

namespace MNN {
	template <typename T>
	class LayerNetwork : public AbstractLayerNetwork<T> {
	private:

	protected:
		NetworkDataContainer<AbstractLayer<T>*> m_layers;
		AbstractLayer<T>* m_inputs;
		AbstractLayer<T>* m_outputs;
	public:
		explicit LayerNetwork(AbstractLayer<T>* inputs, AbstractLayer<T>* outputs) : m_inputs(inputs), m_outputs(outputs) {}
		inline virtual void addLayer(AbstractLayer<T>* l) override {
			m_layers.push_back(l);
		}
		inline virtual void addLayers(const NetworkDataContainer<AbstractLayer<T>*>& c) {
			for (auto t : c)
				this->addLayer(t);
		}
		explicit LayerNetwork(AbstractLayer<T>* inputs, AbstractLayer<T>* outputs, const NetworkDataContainer<AbstractLayer<T>*>& c) : LayerNetwork(inputs, outputs) {
			this->addLayers(c);
		}

		virtual void calculate() override {
			m_outputs->calculate();
		}
	};

	using NeuralNetwork = LayerNetwork<float>;
}