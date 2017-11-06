#pragma once
#include "Storage.hpp"
#include "Layer.hpp"
template<> std::istream& mnn::operator>>(std::istream &s, AbstractLayer<AbstractNeuron> *&res) {
	std::string temp;
	s >> temp;
	if (temp == LayerTypeCode) {
		if (res) delete res;
		res = new Layer<AbstractNeuron>();

		size_t neurons;
		s >> neurons;
		for (size_t i = 0; i < neurons; i++) {
			AbstractNeuron *n = nullptr;
			s >> n;
			res->add(n);
		}
	} else
		throw Exceptions::BrokenMNNFile();
	return s;
}
template<> std::istream& mnn::operator>>(std::istream &s, AbstractLayer<AbstractBackpropagationNeuron> *&res) {
	std::string temp;
	s >> temp;
	if (temp == LayerTypeCode) {
		if (res) delete res;
		res = new Layer<AbstractBackpropagationNeuron>();

		size_t neurons;
		s >> neurons;
		for (size_t i = 0; i < neurons; i++) {
			AbstractNeuron *n = nullptr;
			s >> n;
			res->add(dynamic_cast<AbstractBackpropagationNeuron*>(n));
		}
	} else
		throw Exceptions::BrokenMNNFile();
	return s;
}