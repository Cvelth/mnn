#pragma once
#include "AbstractLayer.hpp"
#include "AbstractNetworkControlFunctions.hpp"
namespace mnn {
	template <typename NeuronType>
	class AbstractInputOutputNetworkStorage : public AbstractMatrixNetworkControlFunctions<NeuronType> {
	protected:
		AbstractLayer<NeuronType>* m_inputs;
		AbstractLayer<NeuronType>* m_outputs;
	public:
		AbstractInputOutputNetworkStorage(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs)
			: m_inputs(inputs), m_outputs(outputs) {}
		AbstractLayer<NeuronType>* inputs() { return m_inputs; }
		AbstractLayer<NeuronType> const* inputs() const { return m_inputs; }
		AbstractLayer<NeuronType>* outputs() { return m_outputs; }
		AbstractLayer<NeuronType> const* outputs() const { return m_outputs; }

		virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_inputs->for_each(lambda, firstToLast);
		}
		virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool const firstToLast = true) const override {
			m_inputs->for_each(lambda, firstToLast);
		}
		virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_outputs->for_each(lambda, firstToLast);
		}
		virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			m_outputs->for_each(lambda, firstToLast);
		}
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) =0;
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const =0;
		virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
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
		virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
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