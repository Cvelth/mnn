#include "Storage.hpp"
#define StorageSystemVersionCode "MNN_v0.2.53_storage_v1.0"
#include <fstream>
#include "AbstractNetwork.hpp"
void mnn::save_to_file(std::string &filename, AbstractNetwork *network) {
	if (network == nullptr)
		throw Exceptions::InvalidNetworkException();
	std::ofstream f;
	f.open(filename, std::ofstream::out);
	f << StorageSystemVersionCode << ' ';
	f << *network;
}
void mnn::load_from_file(std::string &filename, AbstractNetwork *network, bool ignoreUnsupportedMessage) {
	std::ifstream f;
	f.open(filename, std::ifstream::in);
	std::string version;
	f >> version;
	if (!ignoreUnsupportedMessage && version != StorageSystemVersionCode)
		throw Exceptions::UnsupportedMNNFile();
	f >> *network;
}