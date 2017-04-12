#pragma once
#include "NeuronContainer.hpp"
#include "AbstractNeuron.hpp"
#include "Link.hpp"

namespace MNN {
	class AbstractDataContainerNeuron : public AbstractNeuron {
	protected:
		NeuronDataContainer<Link> m_links;
	protected:
		virtual void calculate() override;
		virtual float normalize(const float& value) override;
	public:
		AbstractDataContainerNeuron(const float& value) : AbstractNeuron(value) {}
		AbstractDataContainerNeuron() : AbstractNeuron() {}
		inline virtual void addInput(AbstractNeuron* i, float weight = 1.f) override {
			m_links.push_back(Link(i, weight));
			this->changed();
		}
		inline void addInputs(const NeuronDataContainer<AbstractNeuron*>& c) {
			for (AbstractNeuron* t : c)
				m_links.push_back(Link(t, 1.f));
			this->changed();
		}
		inline void setInputs(const NeuronDataContainer<AbstractNeuron*>& c) {
			m_links.clear();
			m_links.reserve(c.size());
			this->addInputs(c);
		}
		inline AbstractDataContainerNeuron(const NeuronDataContainer<AbstractNeuron*>& c) {
			this->setInputs(c);
		}

		inline virtual void for_each(std::function<void(Link&)> lambda) override {
			for (auto it : m_links)
				lambda(it);
		}
	};

	using Neuron = AbstractDataContainerNeuron;
}