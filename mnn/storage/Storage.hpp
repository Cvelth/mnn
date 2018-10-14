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

		//Management codes.
		error = 0x00,
		separator = 0x01,

		//NeuralNetwork codes.
		explicitly_linked_network = 0x10,
		explicitly_linked_network_backpropagation = 0x11,

		matrix_layered_network = 0x12,
		matrix_layered_network_backpropagation = 0x13,

		//Neuron codes.
		neuron = 0x20,
		neuron_backpropagation = 0x21,

		//Layer codes.
		layer = 0x30,
		layer_backpropagation = 0x31,

		//Link codes.
		link = 0x40,
		link_backpropagation = 0x41,
	};
}