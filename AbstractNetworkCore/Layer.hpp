#pragma once
#include "LayerContainer.hpp"
#include "AbstractLayer.hpp"

namespace MNN {
	class AbstractDataContainerLayer : public AbstractLayer {
	protected:
		LayerDataContainer<AbstractNeuron*> m_neurons;
	public:
		AbstractDataContainerLayer() : AbstractLayer() {}
		inline virtual void add(AbstractNeuron* i) override {
			m_neurons.insert(i);
		}
		inline void addAll(const LayerDataContainer<AbstractNeuron*>& c) {
			for (auto t : c)
				this->add(t);
		}
		inline AbstractDataContainerLayer(const LayerDataContainer<AbstractNeuron*>& c) : AbstractDataContainerLayer() {
			this->addAll(c);
		}

		inline virtual void remove(AbstractNeuron* i) override {
			m_neurons.erase(i);
		}
		inline void removeAll(const LayerDataContainer<AbstractNeuron*>& c) {
			for (auto t : c)
				this->remove(t);
		}

		inline virtual size_t size() const override {
			return m_neurons.size();
		}
		inline virtual void calculate() override {
			for (auto t : m_neurons)
				t->value();
		}
		inline virtual void for_each(std::function<void(AbstractNeuron*)> lambda, bool firstToLast = true) override {
			for (auto it = firstToLast ? m_neurons.begin() : m_neurons.end();
				 it != (firstToLast ? m_neurons.end() : m_neurons.begin());
				 firstToLast ? it++ : it--) {
				lambda(*it);
			}
		}
	};

	using Layer = AbstractDataContainerLayer;
}