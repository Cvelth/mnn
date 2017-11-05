#include "Layer.hpp"
#include <sstream>
std::string mnn::Layer::print() const {
	std::ostringstream res;
	res << "\tL " << m_neurons.size() << '\n';
	for (auto it : m_neurons)
		res << it->print();
	return res.str();
}