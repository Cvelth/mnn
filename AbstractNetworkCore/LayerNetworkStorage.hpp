#pragma once
#include "AbstractInputOutputStorage.hpp"
namespace mnn {
	template <typename NeuronType>
	class LayerNetworkStorage : public AbstractInputOutputNetworkStorage<NeuronType> {
	protected:
		LayerContainer<AbstractLayer<NeuronType>*> m_hidden;
	public:
		LayerNetworkStorage(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs)
			: AbstractInputOutputNetworkStorage(inputs, outputs) {}

		LayerContainer<AbstractLayer<NeuronType>*>& operator*() { return m_hidden; }
		LayerContainer<AbstractLayer<NeuronType>*> const& operator*() const { return m_hidden; }
		LayerContainer<AbstractLayer<NeuronType>*>* operator->() { return &m_hidden; }
		LayerContainer<AbstractLayer<NeuronType>*> const* operator->() const { return &m_hidden; }

		virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) {
			if (firstToLast) for (auto it = m_hidden.begin(); it != m_hidden.end(); it++) lambda(**it);
			else for (auto it = m_hidden.rbegin(); it != m_hidden.rend(); it++) lambda(**it);
		}
		virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const {
			if (firstToLast) for (auto it = m_hidden.begin(); it != m_hidden.end(); it++) lambda(**it);
			else for (auto it = m_hidden.rbegin(); it != m_hidden.rend(); it++) lambda(**it);
		}
		virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) {
			if (firstToLast) {
				lambda(*m_inputs);
				for_each_hidden(lambda, firstToLast);
				lambda(*m_outputs);
			} else {
				lambda(*m_outputs);
				for_each_hidden(lambda, firstToLast);
				lambda(*m_inputs);
			}
		}
		virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const {
			if (firstToLast) {
				lambda(*m_inputs);
				for_each_hidden(lambda, firstToLast);
				lambda(*m_outputs);
			} else {
				lambda(*m_outputs);
				for_each_hidden(lambda, firstToLast);
				lambda(*m_inputs);
			}
		}
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			if (firstToLast) for (auto it = m_hidden.begin(); it != m_hidden.end(); it++) (*it)->for_each(lambda, firstToLast);
			else for (auto it = m_hidden.rbegin(); it != m_hidden.rend(); it++) (*it)->for_each(lambda, firstToLast);
		}
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			if (firstToLast) for (auto it = m_hidden.begin(); it != m_hidden.end(); it++) (*it)->for_each(lambda, firstToLast);
			else for (auto it = m_hidden.rbegin(); it != m_hidden.rend(); it++) (*it)->for_each(lambda, firstToLast);
		}
	};
}