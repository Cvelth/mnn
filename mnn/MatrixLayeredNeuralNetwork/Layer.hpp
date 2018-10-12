#pragma once
#include "mnn/interfaces/Types.hpp"
#include "mnn/exceptions.hpp"
DefineNewMNNException(UnsupportedInputError);
namespace mnn {
	class Layer {
	protected:
		bool m_bias;
		NeuronContainer<NeuronContainer<Value>> m_weights;
	public:
		Layer(size_t const& size, size_t const& input_number, bool bias = true, Value const& minimum_weight_value = 0.0, Value const& maximum_weight_value = 1.0);

		inline size_t size() const { return m_weights.at(0).size(); }
		inline size_t input_number() const { return m_weights.size(); }

		NeuronContainer<Value> process(NeuronContainer<Value> const& inputs);
	};
}