#pragma once
#include "Shared.hpp"
#include "AbstractLayer.hpp"
#include "AbstractLayerNetwork.hpp"
namespace mnn {
	class LayerNetwork : public AbstractLayerNetwork {
	protected:
		AbstractLayer *m_inputs;
		AbstractLayer *m_outputs;
		LayerContainer<AbstractLayer*> m_hidden;
	public:
		explicit LayerNetwork(AbstractLayer *inputs, AbstractLayer *outputs) : m_inputs(inputs), m_outputs(outputs) {}
		explicit LayerNetwork(AbstractLayer *inputs, AbstractLayer *outputs, const LayerContainer<AbstractLayer*>& c)
			: LayerNetwork(inputs, outputs) { addHiddenLayers(c); }
		virtual ~LayerNetwork();
		inline virtual void addHiddenLayer(AbstractLayer *l) override { m_hidden.push_back(l); }
		inline virtual void addHiddenLayers(LayerContainer<AbstractLayer*> const& c) override { for (auto t : c) addHiddenLayer(t); }

		virtual void setInputs(NeuronContainer<Type> const& inputs, bool normalize = true) override;		
		void calculateGradients(const NeuronContainer<Type>& outputs);
		virtual void updateWeights() override;

		virtual NeuronContainer<Type> getInputs() const override;
		virtual NeuronContainer<Type> getOutputs() const override;
		inline AbstractLayer const* getInputs() const { return m_inputs; }
		inline AbstractLayer const* getOutputs() const { return m_outputs; }
		
		inline virtual void calculate() override { m_outputs->calculate(); }
		
		inline virtual void for_each_hidden(std::function<void(AbstractLayer&)> lambda, bool firstToLast = true) override {
			if (firstToLast) for (auto it = m_layers.begin(); it != m_layers.end(); it++) lambda(*it);
			else for (auto it = m_layers.rbegin(); it != m_layers.rend(); it++) lambda(*it);
		}
		inline virtual void for_each_hidden(std::function<void(AbstractLayer const&)> lambda, bool firstToLast = true) const override {
			if (firstToLast) for (auto it = m_layers.begin(); it != m_layers.end(); it++) lambda(*it);
			else for (auto it = m_layers.rbegin(); it != m_layers.rend(); it++) lambda(*it);
		}
		inline virtual void for_each_layer(std::function<void(AbstractLayer&)> lambda, bool firstToLast = true) override {
			if (firstToLast) {
				lambda(m_inputs);
				for_each_hidden(lambda, firstToLast);
				lambda(m_outputs);
			}
			else {
				lambda(m_outputs);
				for_each_hidden(lambda, firstToLast);
				lambda(m_inputs);
			}
		}
		inline virtual void for_each_layer(std::function<void(AbstractLayer const&)> lambda, bool firstToLast = true) const override {
			if (firstToLast) {
				lambda(m_inputs);
				for_each_hidden(lambda, firstToLast);
				lambda(m_outputs);
			}
			else {
				lambda(m_outputs);
				for_each_hidden(lambda, firstToLast);
				lambda(m_inputs);
			}
		}
		inline virtual void for_each_input(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			m_inputs->for_each(lambda, firstToLast);
		}
		inline virtual void for_each_input(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const override {
			m_inputs->for_each(lambda, firstToLast);
		}
		inline virtual void for_each_output(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			m_outputs->for_each(lambda, firstToLast);
		}
		inline virtual void for_each_output(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const override {
			m_outputs->for_each(lambda, firstToLast);
		}
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			if (firstToLast) for (auto it = m_layers.begin(); it != m_layers.end(); it++) (*it)->for_each(lambda, firstToLast);
			else for (auto it = m_layers.rbegin(); it != m_layers.rend(); it++) (*it)->for_each(lambda, firstToLast);
		}
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const override {
			if (firstToLast) for (auto it = m_layers.begin(); it != m_layers.end(); it++) (*it)->for_each(lambda, firstToLast);
			else for (auto it = m_layers.rbegin(); it != m_layers.rend(); it++) (*it)->for_each(lambda, firstToLast);
		}
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			if (firstToLast) {
				m_inputs->for_each(lambda, firstToLast);
				for_each_hidden_neuron(lambda, firstToLast);
				m_outputs->for_each(lambda, firstToLast);
			}
			else {
				m_outputs->for_each(lambda, firstToLast);
				for_each_hidden_neuron(lambda, firstToLast);
				m_inputs->for_each(lambda, firstToLast);
			}
		}
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const override {
			if (firstToLast) {
				m_inputs->for_each(lambda, firstToLast);
				for_each_hidden_neuron(lambda, firstToLast);
				m_outputs->for_each(lambda, firstToLast);
			}
			else {
				m_outputs->for_each(lambda, firstToLast);
				for_each_hidden_neuron(lambda, firstToLast);
				m_inputs->for_each(lambda, firstToLast);
			}
		}
	};
}