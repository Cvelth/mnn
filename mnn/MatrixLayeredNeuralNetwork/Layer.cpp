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
	m_value.resize(size);
}
mnn::BackpropagationLayer::BackpropagationLayer(size_t const& size, size_t const& input_number, bool bias, Value const& minimum_weight_value, Value const& maximum_weight_value)
			: Layer(size, input_number, bias, minimum_weight_value, maximum_weight_value) {
	for (size_t i = 0; i < (bias ? input_number + 1 : input_number); i++) {
		m_deltas.push_back(NeuronContainer<Value>{});
		for (size_t j = 0; j < size; j++)
			m_deltas.back().push_back(Value(0.0));
	}
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

#include "mnn/storage/Storage.hpp"
std::ostream& mnn::Layer::to_stream(std::ostream &output) const {
	output << short(typecodes::layer) << ' ' 
		<< m_weights.size() << ' ' 
		<< m_weights.front().size() << ' ';
	for (auto &row : m_weights)
		for (auto &weight : row)
			output << weight << ' ';
	return output;
}
std::ostream& mnn::BackpropagationLayer::to_stream(std::ostream &output) const {
	output << short(typecodes::layer_backpropagation) << ' '
		<< m_weights.size() << ' '
		<< m_weights.front().size() << ' ';
	for (auto &row : m_weights)
		for (auto &weight : row)
			output << weight << ' ';
	for (auto &row : m_deltas)
		for (auto &delta : row)
			output << delta << ' ';
	return output;
}
std::istream& mnn::Layer::from_stream(std::istream &input) {
	short type;
	input >> type;
	if (type != short(typecodes::layer))
		throw Exceptions::UnsupportedFileError();

	size_t size;
	input >> size;
	m_weights.resize(size);
	input >> size;
	for (auto &row : m_weights)
		row.resize(size);
	for (auto &row : m_weights)
		for (auto &weight : row)
			input >> weight;
	return input;
}
std::istream& mnn::BackpropagationLayer::from_stream(std::istream &input) {
	short type;
	input >> type;
	if (type != short(typecodes::layer_backpropagation))
		throw Exceptions::UnsupportedFileError();

	size_t size;
	input >> size;
	m_weights.resize(size);
	m_deltas.resize(size);
	input >> size;
	for (auto &row : m_weights)
		row.resize(size);
	for (auto &row : m_deltas)
		row.resize(size);
	for (auto &row : m_weights)
		for (auto &weight : row)
			input >> weight;
	for (auto &row : m_deltas)
		for (auto &delta : row)
			input >> delta;
	return input;
}