#pragma once
#include "NeuronContainer.hpp"
#include "AbstractNeuron.hpp"
#include "Link.hpp"

namespace MNN {
	class AbstractLayer;
}
namespace MNN {
	class AbstractDataContainerNeuron : public AbstractNeuron {
	protected:
		NeuronDataContainer<Link> m_links;
	protected:
		virtual void calculate() override;
		virtual float normalize(const float& value) override;
	public:
		AbstractDataContainerNeuron(const float& value, NeuronConstants c = NeuronConstants(0.15f, 0.5f)) : AbstractNeuron(value, c) {}
		AbstractDataContainerNeuron(NeuronConstants c = NeuronConstants(0.15f, 0.5f)) : AbstractNeuron(c) {}
		virtual ~AbstractDataContainerNeuron();
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

		virtual void calculateGradient(float expectedValue) override;
		virtual void calculateGradient(AbstractLayer* nextLayer) override;
		virtual float getWeightTo(AbstractNeuron* neuron) override;
		virtual void recalculateWeights() override;

		inline virtual void for_each(std::function<void(Link&)> lambda, bool firstToLast = true) override {
			if (firstToLast)
				for (auto it = m_links.begin(); it != m_links.end(); it++)
					lambda(*it);
			else
				for (auto it = m_links.rbegin(); it != m_links.rend(); it++)
					lambda(*it);
		}
	};
}