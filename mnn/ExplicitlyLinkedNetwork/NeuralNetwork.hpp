#pragma once
#include <memory>
#include "mnn/interfaces/NeuralNetworkInterface.hpp"
namespace mnn {
	class NeuronInterface;

	class ExplicitlyLinkedNeuralNetwork : public NeuralNetworkInterface {
	protected:
		NeuronContainer<std::shared_ptr<NeuronInterface>> m_hidden;
	public:
		virtual void process() override;
		using NeuralNetworkInterface::process;

		inline void add_input(std::shared_ptr<NeuronInterface> neuron) {
			m_inputs.push_back(neuron);
		}
		inline void add_output(std::shared_ptr<NeuronInterface> neuron) {
			m_outputs.push_back(neuron);
		}
		inline void add_hidden(std::shared_ptr<NeuronInterface> neuron) {
			m_hidden.push_back(neuron);
		}
	};

	class BackpropagationNeuronInterface;

	class ExplicitlyLinkedBackpropagationNeuralNetwork : public BackpropagationNeuralNetworkInterface {
	protected:
		NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>> m_hidden;
	public:
		virtual void process() override;
		using NeuralNetworkInterface::process;

		virtual void calculateGradients(NeuronContainer<Value> const& _outputs) override;
		virtual void calculateGradients(NeuronContainer<std::shared_ptr<NeuronInterface>> const& _outputs) override;
		virtual void updateWeights() override;

		inline void add_input(std::shared_ptr<BackpropagationNeuronInterface> neuron) {
			m_inputs.push_back(neuron);
		}
		inline void add_output(std::shared_ptr<BackpropagationNeuronInterface> neuron) {
			m_outputs.push_back(neuron);
		}
		inline void add_hidden(std::shared_ptr<BackpropagationNeuronInterface> neuron) {
			m_hidden.push_back(neuron);
		}
	};
}