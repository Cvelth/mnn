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
#include "mnn/ExplicitlyLinkedNeuralNetwork/NeuralNetwork.hpp"
#include "mnn/MatrixLayeredNeuralNetwork/NeuralNetwork.hpp" 
std::unique_ptr<mnn::NeuralNetworkInterface> mnn::load_from_file(std::string filename) {
	std::ifstream f(filename, std::ifstream::in);
	if (!f) throw Exceptions::FileAccessError();
	std::string version;
	std::getline(f, version);
	if (version != get_version()) {
		//throw Exceptions::UnsupportedFileError();
	}

	short type;
	size_t inputs, outputs;
	f >> type >> inputs >> outputs;
	switch (typecodes(type)) {
		case typecodes::explicitly_linked_network: {
			auto ret = std::make_unique<ExplicitlyLinkedNeuralNetwork>(inputs, outputs);
			f >> *ret;
			return ret;
		}
		case typecodes::explicitly_linked_network_backpropagation: {
			auto ret = std::make_unique<ExplicitlyLinkedBackpropagationNeuralNetwork>(inputs, outputs);
			f >> *ret;
			return ret;
		}
		case typecodes::matrix_layered_network: {
			auto ret = std::make_unique<MatrixLayeredNeuralNetwork>(inputs, outputs);
			f >> *ret;
			return ret;
		}
		case typecodes::matrix_layered_network_backpropagation: {
			auto ret = std::make_unique<MatrixLayeredBackpropagationNeuralNetwork>(inputs, outputs);
			f >> *ret;
			return ret;
		}
		default:
			throw Exceptions::UnsupportedFileError();
	}
}
