#pragma once
#include "Shared.hpp"
#include "AbstractNeuron.hpp"
#include "AbstractLayer.hpp"
namespace mnn {
	template <typename NeuronType>
	class Layer : public AbstractLayer<NeuronType> {
	protected:
		NeuronContainer<AbstractNeuron*> m_neurons;
	public:
		Layer() : AbstractLayer() {}
		Layer(const NeuronContainer<AbstractNeuron*>& c) : Layer() { addAll(c); }
		virtual ~Layer() { for (auto neuron : m_neurons) delete neuron; }
		inline virtual void add(AbstractNeuron* n) override { m_neurons.push_back(n); }
		inline void addAll(const NeuronContainer<AbstractNeuron*>& c) { for (auto t : c) add(t); }
		inline virtual void clear() { m_neurons.clear(); }
		inline virtual size_t size() const override { return m_neurons.size(); }
		inline virtual Type at(size_t index) const override { return m_neurons.at(index)->value(); }
		virtual std::string print() const override;
		inline virtual void calculate() override { for (auto t : m_neurons) t->value(); }
		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			if (firstToLast)
				for (auto it = m_neurons.begin(); it != m_neurons.end(); it++) lambda(**it);
			else
				for (auto it = m_neurons.rbegin(); it != m_neurons.rend(); it++) lambda(**it);
		}
		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool const firstToLast = true) const override {
			if (firstToLast)
				for (auto it = m_neurons.begin(); it != m_neurons.end(); it++) lambda(**it);
			else
				for (auto it = m_neurons.rbegin(); it != m_neurons.rend(); it++) lambda(**it);
		}
	};
}
#include <sstream>
template <typename NeuronType>
std::string mnn::Layer<NeuronType>::print() const {
	std::ostringstream res;
	res << "\t" << LayerTypeCode << " " << m_neurons.size() << '\n';
	for (auto it : m_neurons)
		res << it->print();
	return res.str();
}
#include "TypeCodes.hpp"
template<typename NeuronType>
std::istream& mnn::operator>>(std::istream &s, AbstractLayer<NeuronType> *&res) {
	std::string temp;
	s >> temp;
	if (temp == LayerTypeCode) {
		if (res) delete res;
		res = new Layer<NeuronType>();

		size_t neurons;
		s >> neurons;
		for (size_t i = 0; i < neurons; i++) {
			NeuronType *n = nullptr;
			s >> n;
			res->add(n);
		}
	} else
		throw Exceptions::BrokenMNNFile();
	return s;
}