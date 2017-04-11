#pragma once
#include "LayerContainer.hpp"
#include "AbstractLayer.hpp"

namespace MNN {
	template <typename T>
	class AbstractDataContainerLayer : public AbstractLayer<T> {
	protected:
		LayerDataContainer<AbstractNeuron<T>*> m_neurons;
	public:
		AbstractDataContainerLayer() : AbstractLayer() {}
		inline virtual void add(AbstractNeuron<T>* i) override {
			m_neurons.insert(i);
		}
		inline void addAll(const LayerDataContainer<AbstractNeuron<T>*>& c) {
			for (auto t : c)
				this->add(t);
		}
		AbstractDataContainerLayer(const LayerDataContainer<AbstractNeuron<T>*>& c) : AbstractDataContainerLayer() {
			this->addAll(c);
		}

		inline virtual void remove(AbstractNeuron<T>* i) override {
			m_neurons.erase(i);
		}
		inline void removeAll(const LayerDataContainer<AbstractNeuron<T>*>& c) {
			for (auto t : c)
				this->remove(t);
		}

		inline virtual void calculate() override {
			for (auto t : m_neurons)
				t->value();
		}
	};

	using Layer = AbstractDataContainerLayer<float>;
}