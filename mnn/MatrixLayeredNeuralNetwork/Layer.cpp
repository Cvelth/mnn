#include "Layer.hpp"
#include <random>
mnn::Layer::Layer(size_t const& size, size_t const& input_number, bool bias, Value const& minimum_weight_value, Value const& maximum_weight_value) : m_bias(bias) {
	static std::mt19937_64 g(std::random_device{}());
	std::uniform_real_distribution<> d(minimum_weight_value, maximum_weight_value);
	for (size_t i = 0; i < (bias ? input_number + 1 : input_number); i++) {
		m_weights.push_back(NeuronContainer<Value>{});
		for (size_t j = 0; j < size; j++)
			m_weights.back().push_back(d(g));
	}
}

mnn::NeuronContainer<mnn::Value> mnn::Layer::process(NeuronContainer<Value> const& inputs) {
	if (inputs.size() != m_weights.size() && (!m_bias || inputs.size() + 1 != m_weights.size()))
		throw Exceptions::UnsupportedInputError();

	mnn::NeuronContainer<mnn::Value> res;
	for (size_t i = 0; i < m_weights.front().size(); i++) {
		Value temp = 0;
		for (size_t j = 0; j < (m_bias ? m_weights.size() - 1 : m_weights.size()); j++)
			temp += inputs[j] * m_weights[j][i];
		if (m_bias)
			temp += m_weights.back()[i];
		res.push_back(normalize(temp));
	}

	return res;
}
