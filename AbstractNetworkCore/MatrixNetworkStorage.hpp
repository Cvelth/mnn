#pragma once
#include "AbstractInputOutputStorage.hpp"
namespace mnn {
	template <typename NeuronType>
	class MatrixNetworkStorage : public AbstractInputOutputNetworkStorage<NeuronType> {
	protected:
		Matrix<AbstractNeuron> m_matrix;
	public:
		MatrixNetworkStorage(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs)
			: AbstractInputOutputNetworkStorage(inputs, outputs) {}

		Matrix<AbstractNeuron>& operator*() { return m_matrix; }
		Matrix<AbstractNeuron> const& operator*() const { return m_matrix; }
		Matrix<AbstractNeuron>* operator->() { return &m_matrix; }
		Matrix<AbstractNeuron> const* operator->() const { return &m_matrix; }

		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_matrix->for_each(lambda, firstToLast);
		}
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			m_matrix->for_each(lambda, firstToLast);
		}
	};
}