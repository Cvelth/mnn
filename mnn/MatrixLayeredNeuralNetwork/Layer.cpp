#include "Layer.hpp"
#include <random>
mnn::Layer::Layer(size_t const& size, size_t const& input_number, bool bias, Value const& minimum_weight_value, Value const& maximum_weight_value) : m_bias(bias) {
	static std::mt19937_64 g(std::random_device{}());
	std::uniform_real_distribution<> d(minimum_weight_value, maximum_weight_value);

	for (size_t i = 0; i < (bias ? input_number + 1 : input_number); i++) {
		m_weights.push_back(NeuronContainer<Value>{});
		m_deltas.push_back(NeuronContainer<Value>{});
		for (size_t j = 0; j < size; j++) {
			m_weights.back().push_back(d(g));
			m_deltas.back().push_back(0.0);
		}
	}
	m_value.resize(size);
}

mnn::NeuronContainer<mnn::Value> mnn::Layer::process(NeuronContainer<Value> const& inputs) {
	if (inputs.size() != m_weights.size() && (!m_bias || inputs.size() + 1 != m_weights.size()))
		throw Exceptions::UnsupportedInputError();

	m_value.clear();
	for (size_t i = 0; i < m_weights.front().size(); i++) {
		m_value.push_back(0.0);
		for (size_t j = 0; j < (m_bias ? m_weights.size() - 1 : m_weights.size()); j++)
			m_value.at(i) += inputs[j] * m_weights[j][i];
		if (m_bias)
			m_value.at(i) += m_weights.back()[i];
		m_value.at(i) = normalize(m_value.at(i));
	}
	return m_value;
}