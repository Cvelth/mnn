#pragma once
#include <string>
namespace mnn {
	class AbstractNetwork;
	void save_to_file(std::string &filename, AbstractNetwork *network);
	void load_from_file(std::string &filename, AbstractNetwork *network);
}