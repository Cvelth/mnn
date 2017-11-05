#include "Storage.hpp"
#define StorageSystemVersionCode "MNN_v0.2.53_storage_v1.0"
#include <fstream>
#include "AbstractNetwork.hpp"
void mnn::save_to_file(std::string filename, AbstractNetwork *network) {
	if (network == nullptr)
		throw Exceptions::InvalidNetworkException();
	std::ofstream f;
	f.open(filename, std::ofstream::out);
	f << StorageSystemVersionCode << '\n';
	f << network;
}
void mnn::load_from_file(std::string filename, AbstractNetwork *network, bool ignoreUnsupportedMessage) {
	std::ifstream f;
	f.open(filename, std::ifstream::in);
	std::string version;
	f >> version;
	if (!ignoreUnsupportedMessage && version != StorageSystemVersionCode)
		throw Exceptions::UnsupportedMNNFile();
	f >> network;
}
std::ostream& mnn::operator<<(std::ostream &s, AbstractNetwork const* n) {
	if (n) return s << n->print();
	else throw Exceptions::NonExistingNetworkUsed();
}
std::istream& mnn::operator>>(std::istream &s, AbstractNetwork *n) {
	std::string t;
	s >> t;
	//n = new
	return s;
}