#include "Storage.hpp"
#include <fstream>
#include "AbstractNetwork.hpp"
#include "TypeCodes.hpp"
void mnn::save_to_file(std::string filename, AbstractNetwork *network) {
	if (network == nullptr)
		throw Exceptions::InvalidNetworkException();
	std::ofstream f;
	f.open(filename, std::ofstream::out);
	f << StorageSystemVersionCode << '\n';
	f << network;
}
std::ostream& mnn::operator<<(std::ostream &s, AbstractNetwork const* n) {
	if (n) return s << n->print();
	else throw Exceptions::NonExistingNetworkUsed();
}
#include "AbstractNeuron.hpp"
#include <map>
std::map<size_t, std::pair<mnn::AbstractNeuron*, bool>> neuron_map;
mnn::AbstractNetwork* mnn::load_from_file(std::string filename, bool ignoreUnsupportedMessage) {
	std::ifstream f;
	f.open(filename, std::ifstream::in);
	std::string version;
	f >> version;
	if (!ignoreUnsupportedMessage && version != StorageSystemVersionCode)
		throw Exceptions::UnsupportedMNNFile();
	mnn::AbstractNetwork *res = nullptr;
	f >> res;
	for (auto it : neuron_map)
		if (!it.second.second)
			it.second.first->setValueUnnormalized(1.f);
	return res;
}
#include "LayerNetwork.hpp"
std::istream& mnn::operator>>(std::istream &s, AbstractNetwork *&res) {
	neuron_map.clear();
	std::string temp;
	s >> temp;
	if (temp == LayerNetworkTypeCode) {
		s >> temp;
		if (temp != InputsTypeCode)
			throw Exceptions::BrokenMNNFile();
		AbstractLayer<AbstractNeuron> *inputs = nullptr;
		s >> inputs;

		s >> temp;
		if (temp != OutputsTypeCode)
			throw Exceptions::BrokenMNNFile();
		AbstractLayer<AbstractNeuron> *outputs = nullptr;
		s >> outputs;

		s >> temp;
		if (temp != HiddenTypeCode)
			throw Exceptions::BrokenMNNFile();
		LayerContainer<AbstractLayer<AbstractNeuron>*> hidden;
		size_t hidden_number;
		s >> hidden_number;
		for (size_t i = 0; i < hidden_number; i++) {
			AbstractLayer<AbstractNeuron> *hidden_temp = nullptr;
			s >> hidden_temp;
			hidden.push_back(hidden_temp);
		}

		s >> temp;
		if (temp != LayerNetworkTypeCode)
			throw Exceptions::BrokenMNNFile();

		res = new LayerNetwork(inputs, outputs, hidden);
	} else if (temp == BackpropagationLayerNetworkTypeCode) {
		s >> temp;
		if (temp != InputsTypeCode)
			throw Exceptions::BrokenMNNFile();
		AbstractLayer<AbstractBackpropagationNeuron> *inputs = nullptr;
		s >> inputs;

		s >> temp;
		if (temp != OutputsTypeCode)
			throw Exceptions::BrokenMNNFile();
		AbstractLayer<AbstractBackpropagationNeuron> *outputs = nullptr;
		s >> outputs;

		s >> temp;
		if (temp != HiddenTypeCode)
			throw Exceptions::BrokenMNNFile();
		LayerContainer<AbstractLayer<AbstractBackpropagationNeuron>*> hidden;
		size_t hidden_number;
		s >> hidden_number;
		for (size_t i = 0; i < hidden_number; i++) {
			AbstractLayer<AbstractBackpropagationNeuron> *hidden_temp = nullptr;
			s >> hidden_temp;
			hidden.push_back(hidden_temp);
		}

		s >> temp;
		if (temp != LayerNetworkTypeCode)
			throw Exceptions::BrokenMNNFile();

		res = new BackpropagationLayerNetwork(inputs, outputs, hidden);
	} else
		throw Exceptions::BrokenMNNFile();
	return s;
}
#include "Neuron.hpp"
std::istream& mnn::operator>>(std::istream &s, AbstractNeuron *&res) {
	std::string temp;
	s >> temp;
	if (temp == NeuronTypeCode) {
		size_t id, links;
		s >> id;
		if (res) delete res;
		res = new Neuron();
		s >> links;
		for (size_t i = 0; i < links; i++) {
			s >> temp;
			if (temp == LinkTypeCode) {
				size_t id;
				Type weight;
				s >> id >> weight;
				mnn::AbstractNeuron *tn = nullptr;
				auto it = neuron_map.find(id);
				if (it == neuron_map.end())
					neuron_map.insert(std::make_pair(id, std::make_pair(tn = new mnn::Neuron(), false)));
				else
					tn = it->second.first;
				res->link(Link(tn, weight));
			} else 
				throw Exceptions::BrokenMNNFile();
		}
		
		auto it = neuron_map.find(id);
		if (it == neuron_map.end()) 
			neuron_map.insert(std::make_pair(id, std::make_pair(res, true)));
		else {
			it->second.second = true;
			it->second.first->update_links(res->links());
			res = it->second.first;
		}
	} else if (temp == BackpropagationNeuronTypeCode) {
		size_t id, links;
		Type eta, alpha;
		s >> id >> eta >> alpha;
		if (res) delete res;
		res = new BackpropagationNeuron(eta, alpha);
		s >> links;
		for (size_t i = 0; i < links; i++) {
			s >> temp;
			if (temp == BackpropagationLinkTypeCode) {
				size_t id;
				Type weight, delta;
				s >> id >> weight >> delta;
				mnn::AbstractBackpropagationNeuron *tn = nullptr;
				auto it = neuron_map.find(id);
				if (it == neuron_map.end())
					neuron_map.insert(std::make_pair(id, std::make_pair(tn = new mnn::BackpropagationNeuron(0.f, 0.f), false)));
				else
					tn = dynamic_cast<AbstractBackpropagationNeuron*>(it->second.first);
				res->link(BackpropagationLink(tn, weight, delta));
			} else
				throw Exceptions::BrokenMNNFile();
		}

		auto it = neuron_map.find(id);
		if (it == neuron_map.end())
			neuron_map.insert(std::make_pair(id, std::make_pair(res, true)));
		else {
			it->second.second = true; 
			auto temp_res = dynamic_cast<AbstractBackpropagationNeuron*>(res);
			auto temp_map = dynamic_cast<AbstractBackpropagationNeuron*>(it->second.first);
			temp_map->m_alpha = temp_res->m_alpha;
			temp_map->m_eta = temp_res->m_eta;
			temp_map->update_links(temp_res->links());
			res = temp_map;
		}
	} else
		throw Exceptions::BrokenMNNFile();
	return s;
}