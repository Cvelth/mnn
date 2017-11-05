#pragma once
#include "AbstractLayer.hpp"
namespace mnn {
	template <typename NeuronType>
	class LayerNetworkStorage {
	protected:
		AbstractLayer<NeuronType>* m_inputs;
		AbstractLayer<NeuronType>* m_outputs;
		LayerContainer<AbstractLayer<NeuronType>*> m_hidden;
	public:
		LayerNetworkStorage(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs)
			: m_inputs(inputs), m_outputs(outputs) {}
		AbstractLayer<NeuronType>* inputs() { return m_inputs; }
		AbstractLayer<NeuronType> const* inputs() const { return m_inputs; }
		AbstractLayer<NeuronType>* outputs() { return m_outputs; }
		AbstractLayer<NeuronType> const* outputs() const { return m_outputs; }

		LayerContainer<AbstractLayer<NeuronType>*>& operator*() { return m_hidden; }
		LayerContainer<AbstractLayer<NeuronType>*> const& operator*() const { return m_hidden; }
		LayerContainer<AbstractLayer<NeuronType>*>* operator->() { return &m_hidden; }
		LayerContainer<AbstractLayer<NeuronType>*> const* operator->() const { return &m_hidden; }

		inline void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) {
			if (firstToLast) for (auto it = m_hidden.begin(); it != m_hidden.end(); it++) lambda(**it);
			else for (auto it = m_hidden.rbegin(); it != m_hidden.rend(); it++) lambda(**it);
		}
		inline void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const {
			if (firstToLast) for (auto it = m_hidden.begin(); it != m_hidden.end(); it++) lambda(**it);
			else for (auto it = m_hidden.rbegin(); it != m_hidden.rend(); it++) lambda(**it);
		}
		inline void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) {
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
		inline void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const {
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
		inline void for_each_input(std::function<void(NeuronType&)> lambda, bool firstToLast = true) {
			m_inputs->for_each(lambda, firstToLast);
		}
		inline void for_each_input(std::function<void(NeuronType&)> lambda, bool const firstToLast = true) const {
			m_inputs->for_each(lambda, firstToLast);
		}
		inline void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) {
			m_outputs->for_each(lambda, firstToLast);
		}
		inline void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const {
			m_outputs->for_each(lambda, firstToLast);
		}
		inline void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) {
			if (firstToLast) for (auto it = m_hidden.begin(); it != m_hidden.end(); it++) (*it)->for_each(lambda, firstToLast);
			else for (auto it = m_hidden.rbegin(); it != m_hidden.rend(); it++) (*it)->for_each(lambda, firstToLast);
		}
		inline void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const {
			if (firstToLast) for (auto it = m_hidden.begin(); it != m_hidden.end(); it++) (*it)->for_each(lambda, firstToLast);
			else for (auto it = m_hidden.rbegin(); it != m_hidden.rend(); it++) (*it)->for_each(lambda, firstToLast);
		}
		inline void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) {
			if (firstToLast) {
				m_inputs->for_each(lambda, firstToLast);
				for_each_hidden_neuron(lambda, firstToLast);
				m_outputs->for_each(lambda, firstToLast);
			} else {
				m_outputs->for_each(lambda, firstToLast);
				for_each_hidden_neuron(lambda, firstToLast);
				m_inputs->for_each(lambda, firstToLast);
			}
		}
		inline void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const {
			if (firstToLast) {
				m_inputs->for_each(lambda, firstToLast);
				for_each_hidden_neuron(lambda, firstToLast);
				m_outputs->for_each(lambda, firstToLast);
			} else {
				m_outputs->for_each(lambda, firstToLast);
				for_each_hidden_neuron(lambda, firstToLast);
				m_inputs->for_each(lambda, firstToLast);
			}
		}
	};
}