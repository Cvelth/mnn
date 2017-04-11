#pragma once
#include "NeuronContainer.hpp"
#include "AbstractNeuron.hpp"
#include "Link.hpp"

namespace MNN {
	template <typename T>
	class AbstractDataContainerNeuron : public AbstractNeuron<T> {
	protected:
		NeuronDataContainer<Link<T>> m_links;
	protected:
		virtual void calculate() override {
			T value = T(0);
			for (Link<T> t : m_links)
				value += t.unit->value() * t.weight;
			this->setValue(value);
		};
		virtual T normalize(const T& value) override {
			return value; //Does Nothing
		}
	public:
		AbstractDataContainerNeuron(const T& value) : AbstractNeuron(value) {}
		AbstractDataContainerNeuron() : AbstractNeuron() {}
		inline virtual void addInput(AbstractNeuron<T>* i) override {
			m_links.push_back(Link<T>(i, 1.f));
			this->changed();
		}
		inline void addInputs(const NeuronDataContainer<AbstractNeuron<T>*>& c) {
			for (AbstractNeuron<T>* t : c)
				m_links.push_back(Link<T>(t, 1.f));
			this->changed();
		}
		inline void setInputs(const NeuronDataContainer<AbstractNeuron<T>*>& c) {
			m_links.clear();
			m_links.reserve(c.size());
			this->addInputs(c);
		}
		AbstractDataContainerNeuron(const NeuronDataContainer<AbstractNeuron<T>*>& c) {
			this->setInputs(c);
		}
	};

	using Neuron = AbstractDataContainerNeuron<float>;
}