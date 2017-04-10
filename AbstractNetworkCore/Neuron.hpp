#pragma once
#include "Container.hpp"
#include "AbstractNeuron.hpp"
#include "Link.hpp"

namespace MNN {
	template <typename T>
	class VectorNeuron : public AbstractNeuron<T> {
	protected:
		Container<Link<T>> m_links;
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
		using AbstractNeuron<T>::AbstractNeuron;
		inline void setInputs(const Container<AbstractNeuron<T>*>& c) {
			m_links.clear();
			m_links.reserve(c.size());
			this->addInputs(c);
			this->changed();
		}
		inline void addInputs(const Container<AbstractNeuron<T>*>& c) {
			for (AbstractNeuron<T>* t : c)
				this->addInput(t);
		}
		inline void addInput(AbstractNeuron<T>* i) {
			m_links.push_back(Link<T>(i, 1.f));
		}
	};

	using Neuron = VectorNeuron<float>;
}