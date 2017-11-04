#pragma once
#include "Shared.hpp"
#include <string>
namespace mnn {
	GenerateNewException(InvalidNetworkException);
	GenerateNewException(UnsupportedMNNFile);
	class AbstractNetwork;
	void save_to_file(std::string filename, AbstractNetwork *network);
	void load_from_file(std::string filename, AbstractNetwork *network, bool ignoreUnsupportedMessage = false);
}