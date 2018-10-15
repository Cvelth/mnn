#pragma once
#include <memory>
#include "mnn/interfaces/NeuralNetworkInterface.hpp"
#include "mnn/exceptions.hpp"
DefineNewMNNException(MatrixStructureIsBroken);
namespace mnn {
	class Layer;
	class MatrixLayeredNeuralNetwork : public NeuralNetworkInterface {
	protected:
		LayerContainer<std::shared_ptr<Layer>> m_layers;
	protected:
		virtual std::ostream& to_stream(std::ostream &output) const override;
		virtual std::istream& from_stream(std::istream &input) override;
	public:
		using NeuralNetworkInterface::NeuralNetworkInterface;
		virtual void process() override;
		using NeuralNetworkInterface::process;

		inline LayerContainer<std::shared_ptr<Layer>> layers() { return m_layers; }
		inline LayerContainer<std::shared_ptr<Layer>> const& layers() const { return m_layers; }

		void add_layer(size_t const& size, bool bias = true, Value const& minimum_weight_value = 0.0, Value const& maximum_weight_value = 1.0);
	};

	class BackpropagationLayer;
	class MatrixLayeredBackpropagationNeuralNetwork : public BackpropagationNeuralNetworkInterface {
	protected:
		Value m_eta, m_alpha;
		LayerContainer<std::shared_ptr<BackpropagationLayer>> m_layers;
	protected:
		virtual std::ostream& to_stream(std::ostream &output) const override;
		virtual std::istream& from_stream(std::istream &input) override;
	public:
		MatrixLayeredBackpropagationNeuralNetwork(size_t const& input_number, size_t const& output_number, Value const& eta = 0.15, Value const& alpha = 0.5);
		virtual void process() override;
		using BackpropagationNeuralNetworkInterface::process;

		virtual void backpropagate(NeuronContainer<Value> const& _outputs) override;

		inline LayerContainer<std::shared_ptr<BackpropagationLayer>> layers() { return m_layers; }
		inline LayerContainer<std::shared_ptr<BackpropagationLayer>> const& layers() const { return m_layers; }

		void add_layer(size_t const& size, bool bias = true, Value const& minimum_weight_value = 0.0, Value const& maximum_weight_value = 1.0);
	};
}