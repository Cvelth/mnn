#pragma once
#include "Shared.hpp"
#include <string>
namespace mnn {
	GenerateNewException(InvalidNetworkException);
	GenerateNewException(CannotAccessFile);
	GenerateNewException(UnsupportedMNNFile);
	GenerateNewException(BrokenMNNFile);
	class AbstractNetwork;
	void save_to_file(std::string filename, AbstractNetwork *network);
	AbstractNetwork* load_from_file(std::string filename, bool ignoreUnsupportedMessage = false);
}