#pragma once
#include <memory>
#include "mnn/interfaces/NeuralNetworkInterface.hpp"
#include "mnn/exceptions.hpp"
DefineNewMNNException(IndexIsTooLargeError);
namespace mnn {
	class NeuronInterface;

	class ExplicitlyLinkedNeuralNetwork : public NeuralNetworkInterface {
	protected:
		NeuronContainer<std::shared_ptr<NeuronInterface>> m_input_neurons;
		NeuronContainer<std::shared_ptr<NeuronInterface>> m_output_neurons;
		NeuronContainer<std::shared_ptr<NeuronInterface>> m_hidden_neurons;
	protected:
		virtual std::ostream& to_stream(std::ostream &output) const override;
		virtual std::istream& from_stream(std::istream &input) override;
	public:
		ExplicitlyLinkedNeuralNetwork(size_t input_number, size_t output_number);
		virtual void process() override;
		using NeuralNetworkInterface::process;

		inline NeuronContainer<std::shared_ptr<NeuronInterface>>& hidden_neurons() { return m_hidden_neurons; }
		inline NeuronContainer<std::shared_ptr<NeuronInterface>> const& hidden_neurons() const { return m_hidden_neurons; }
		
		inline NeuronContainer<std::shared_ptr<NeuronInterface>>& input_neurons() { return m_input_neurons; }
		inline NeuronContainer<std::shared_ptr<NeuronInterface>> const& input_neurons() const { return m_input_neurons; }
		
		inline NeuronContainer<std::shared_ptr<NeuronInterface>>& output_neurons() { return m_output_neurons; }
		inline NeuronContainer<std::shared_ptr<NeuronInterface>> const& output_neurons() const { return m_output_neurons; }
	
		static std::shared_ptr<ExplicitlyLinkedNeuralNetwork> generate(ExplicitlyLinkedNeuralNetwork const&, ExplicitlyLinkedNeuralNetwork const&, Value const& ratio = Value(0.5));
	};

	class BackpropagationNeuronInterface;

	class ExplicitlyLinkedBackpropagationNeuralNetwork : public BackpropagationNeuralNetworkInterface {
	protected:
		NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>> m_input_neurons;
		NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>> m_output_neurons;
		NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>> m_hidden_neurons;
	protected:
		virtual std::ostream& to_stream(std::ostream &output) const override;
		virtual std::istream& from_stream(std::istream &input) override;
	public:
		ExplicitlyLinkedBackpropagationNeuralNetwork(size_t input_number, size_t output_number);
		virtual void process() override;
		using NeuralNetworkInterface::process;

		virtual void backpropagate(NeuronContainer<Value> const& _outputs) override;

		inline NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>>& hidden_neurons() { return m_hidden_neurons; }
		inline NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>> const& hidden_neurons() const { return m_hidden_neurons; }

		inline NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>>& input_neurons() { return m_input_neurons; }
		inline NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>> const& input_neurons() const { return m_input_neurons; }

		inline NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>>& output_neurons() { return m_output_neurons; }
		inline NeuronContainer<std::shared_ptr<BackpropagationNeuronInterface>> const& output_neurons() const { return m_output_neurons; }
	};
}