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
	public:
		using NeuralNetworkInterface::NeuralNetworkInterface;
		virtual void process() override;
		using NeuralNetworkInterface::process;

		inline LayerContainer<std::shared_ptr<Layer>> layers() { return m_layers; }
		inline LayerContainer<std::shared_ptr<Layer>> const& layers() const { return m_layers; }

		void add_layer(size_t const& size, bool bias = true, Value const& minimum_weight_value = 0.0, Value const& maximum_weight_value = 1.0);
	};
}