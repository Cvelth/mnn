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

mnn::Layer::Layer(size_t const& size, size_t const& input_number, bool bias, std::function<Value(size_t, size_t)> const& weight_function) : m_bias(bias) {
	for (size_t i = 0; i < (bias ? input_number + 1 : input_number); i++) {
		m_weights.push_back(NeuronContainer<Value>{});
		for (size_t j = 0; j < size; j++)
			m_weights.back().push_back(weight_function(j, i));
	}
	m_value.resize(size);
}
mnn::BackpropagationLayer::BackpropagationLayer(size_t const& size, size_t const& input_number, bool bias, std::function<Value(size_t, size_t)> const& weight_function)
			: Layer(size, input_number, bias, weight_function) {
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

size_t is_inside(size_t const& v, size_t const& b1, size_t const& b2) {
	if (v < b1)
		if (v < b2)
			return 0b11;
		else
			return 0b01;
	else
		if (v < b2)
			return 0b10;
		else
			return 0b00;
};
std::shared_ptr<mnn::Layer> mnn::Layer::generate(size_t const& input_number, Layer const& l1, Layer const& l2, Value const& ratio, Value const& d) {
	static std::mt19937_64 g(std::random_device{}());
	std::bernoulli_distribution b(ratio);
	
	auto ret = std::make_shared<Layer>(
		b(g) ? l1.size() : l2.size(),
		input_number,
		b(g) ? l1.m_bias : l2.m_bias,
		[&l1, &l2, &b, &d](size_t const& j, size_t const& i) -> Value {
			switch (is_inside(i, l1.input_number(), l2.input_number())
					& is_inside(j, l1.size(), l2.size())) {
				case 0b11:
					return b(g) ? l1.m_weights.at(i).at(j) : l2.m_weights.at(i).at(j);
				case 0b10:
					return l2.m_weights.at(i).at(j);
				case 0b01:
					return l1.m_weights.at(i).at(j);
				default:
					return d;
			}
		}
	);

	return ret;
}

std::shared_ptr<mnn::Layer> mnn::Layer::generate(size_t const& input_number, Layer const& l, Value const& d) {
	auto ret = std::make_shared<Layer>(
		l.size(),
		input_number,
		l.m_bias,
		[&l, &d](size_t const& i, size_t const& j) -> Value {
			if (i < l.input_number() && j < l.size())
				return l.m_weights.at(i).at(j);
			else
				return d;
		}
	);

	return ret;
}

void mnn::Layer::for_each_weight(std::function<void(Value&)> lambda) {
	for (auto &row : m_weights)
		for (auto &weight : row)
			lambda(weight);
}

#include "mnn/storage/Storage.hpp"
std::ostream& mnn::Layer::to_stream(std::ostream &output) const {
	output << short(typecodes::layer) << ' ' << m_bias 
		<< m_weights.size() << ' ' 
		<< m_weights.front().size() << ' ';
	for (auto &row : m_weights)
		for (auto &weight : row)
			output << weight << ' ';
	return output;
}
std::ostream& mnn::BackpropagationLayer::to_stream(std::ostream &output) const {
	output << short(typecodes::layer_backpropagation) << ' ' << m_bias
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

	input >> m_bias;

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

	input >> m_bias;

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