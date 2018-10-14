#pragma once
#include <string>
#include "mnn/exceptions.hpp"
DefineNewMNNException(FileAccessError);
DefineNewMNNException(UnsupportedFileError);
namespace mnn {
	class NeuralNetworkInterface;
	void save_to_file(std::string filename, NeuralNetworkInterface const& network);
	std::unique_ptr<NeuralNetworkInterface> load_from_file(std::string filename);
}

namespace mnn {
	enum class typecodes {
		error = 0x00,

		//NeuralNetwork codes.
		explicitly_linked_network = 0x02,
		explicitly_linked_network_backpropagation = 0x03,

		matrix_layered_network = 0x04,
		matrix_layered_network_backpropagation = 0x05,

		//Neuron codes.
		neuron = 0x12,
		neuron_backpropagation = 0x13,

		//Layer codes.
		layer = 0x22,
		layer_backpropagation = 0x23,
	};
}