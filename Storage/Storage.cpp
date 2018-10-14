#include "Storage.hpp"
#include "mnn/version.hpp"
#include <fstream>
#include "mnn/interfaces/NeuralNetworkInterface.hpp"
void mnn::save_to_file(std::string filename, NeuralNetworkInterface const& network) {
	std::ofstream f(filename, std::ofstream::out);
	if (!f) throw Exceptions::FileAccessError();
	f << "MNN " << get_version() << '\n';
	f << network;
}
#include "mnn/ExplicitlyLinkedNetwork/NeuralNetwork.hpp"
#include "mnn/MatrixLayeredNeuralNetwork/NeuralNetwork.hpp" 
std::unique_ptr<mnn::NeuralNetworkInterface> mnn::load_from_file(std::string filename) {
	std::ifstream f(filename, std::ifstream::in);
	if (!f) throw Exceptions::FileAccessError();
	std::string version;
	f >> version;
	if (version != get_version()) {
		throw Exceptions::UnsupportedFileError();
	}

	short type;
	f >> type;
	switch (typecodes(type)) {
		case typecodes::explicitly_linked_network:
			 return std::make_unique<ExplicitlyLinkedNeuralNetwork>(f);
		case typecodes::explicitly_linked_network_backpropagation:
			return std::make_unique<ExplicitlyLinkedBackpropagationNeuralNetwork>(f);
		case typecodes::matrix_layered_network:
			return std::make_unique<MatrixLayeredNeuralNetwork>(f);
		case typecodes::matrix_layered_network_backpropagation:
			return std::make_unique<MatrixLayeredBackpropagationNeuralNetwork>(f);
		default:
			throw Exceptions::UnsupportedFileError();
	}
}
