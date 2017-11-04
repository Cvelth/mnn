#pragma once
#include "Shared.hpp"
#include "AbstractNeuron.hpp"
#include "AbstractLayer.hpp"
namespace mnn {
	class Layer : public AbstractLayer {
	protected:
		NeuronContainer<AbstractNeuron*> m_neurons;
	public:
		Layer() : AbstractLayer() {}
		Layer(const NeuronContainer<AbstractNeuron*>& c) : Layer() { addAll(c); }
		virtual ~Layer() { for (auto neuron : m_neurons) delete neuron; }
		inline virtual void add(AbstractNeuron* n) override { m_neurons.push_back(n); }
		inline void addAll(const NeuronContainer<AbstractNeuron*>& c) { for (auto t : c) add(t); }
		inline virtual void clear() { m_neurons.clear(); }
		inline virtual size_t size() const override { return m_neurons.size(); }
		inline virtual void calculate() override { for (auto t : m_neurons) t->value(); }
		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			if (firstToLast)
				for (auto it = m_neurons.begin(); it != m_neurons.end(); it++) lambda(**it);
			else
				for (auto it = m_neurons.rbegin(); it != m_neurons.rend(); it++) lambda(**it);
		}
		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool const firstToLast = true) const override {
			if (firstToLast)
				for (auto it = m_neurons.begin(); it != m_neurons.end(); it++) lambda(**it);
			else
				for (auto it = m_neurons.rbegin(); it != m_neurons.rend(); it++) lambda(**it);
		}
	};
}