#pragma once
#include "LayerContainer.hpp"
#include "AbstractLayer.hpp"
#include "AbstractDataContainerNeuron.hpp"

namespace mnn {
	class AbstractDataContainerLayer : public AbstractLayer {
	protected:
		LayerDataContainer<AbstractNeuron*> m_neurons;
	public:
		AbstractDataContainerLayer() : AbstractLayer() {}
		virtual ~AbstractDataContainerLayer();
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
			if (firstToLast)
				for (auto it = m_neurons.begin(); it != m_neurons.end(); it++)
					lambda(*it);
			else
				for (auto it = m_neurons.rbegin(); it != m_neurons.rend(); it++)
					lambda(*it);
		}
	};

	using Layer = AbstractDataContainerLayer;
}