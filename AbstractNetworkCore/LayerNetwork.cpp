#include "LayerNetwork.hpp"
#include "TypeCodes.hpp"
#include <sstream>
std::string mnn::LayerNetwork::print() const {
	std::ostringstream res;
	res << LayerNetworkTypeCode << '\n';
	res << InputsTypeCode << " " << m_layers.inputs()->print();
	res << OutputsTypeCode << " " << m_layers.outputs()->print();
	res << HiddenTypeCode << " " << m_layers->size() << '\n';
	for (auto& it : *m_layers)
		res << it->print() << '\n';
	res << LayerNetworkTypeCode;
	return res.str();
}
std::string mnn::BackpropagationLayerNetwork::print() const {
	std::ostringstream res;
	res << BackpropagationLayerNetworkTypeCode << '\n';
	res << InputsTypeCode << " " << m_layers.inputs()->print();
	res << OutputsTypeCode << " " << m_layers.outputs()->print();
	res << HiddenTypeCode << " " << m_layers->size() << '\n';
	for (auto& it : *m_layers)
		res << it->print() << '\n';
	res << BackpropagationLayerNetworkTypeCode;
	return res.str();
}