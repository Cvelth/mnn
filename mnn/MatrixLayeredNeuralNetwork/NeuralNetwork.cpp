#include "NeuralNetwork.hpp"
#include "layer.hpp"
void mnn::MatrixLayeredNeuralNetwork::process() {
	if (m_layers.back()->size() != m_outputs.size())
		throw Exceptions::MatrixStructureIsBroken("Size of the last layer must be equal to number of outputs.");
	
	auto current = m_inputs;
	for (auto &layer : m_layers)
		current = layer->process(current);
	
	auto it1 = m_outputs.begin();
	auto it2 = current.cbegin();
	while (it1 != m_outputs.end() || it2 != current.cend()) {
		*it1 = *it2;
		it1++; it2++;
	}
}
void mnn::MatrixLayeredNeuralNetwork::add_layer(size_t const& size, bool bias, Value const& minimum_weight_value, Value const& maximum_weight_value) {
	m_layers.push_back(std::make_shared<Layer>(size, m_layers.empty() ? m_inputs.size() : m_layers.back()->size(), bias, minimum_weight_value, maximum_weight_value));
}
void mnn::MatrixLayeredNeuralNetwork::add_layer(size_t const& size, bool bias, std::function<Value(size_t, size_t)> const& weight_function) {
	m_layers.push_back(std::make_shared<Layer>(size, m_layers.empty() ? m_inputs.size() : m_layers.back()->size(), bias, weight_function));
}

mnn::MatrixLayeredBackpropagationNeuralNetwork::MatrixLayeredBackpropagationNeuralNetwork(size_t const& input_number, size_t const& output_number, 
																						  Value const& eta, Value const& alpha)
	: BackpropagationNeuralNetworkInterface(input_number, output_number), m_eta(eta), m_alpha(alpha) {}

void mnn::MatrixLayeredBackpropagationNeuralNetwork::process() {
	if (m_layers.back()->size() != m_outputs.size())
		throw Exceptions::MatrixStructureIsBroken("Size of the last layer must be equal to number of outputs.");

	auto current = m_inputs;
	for (auto &layer : m_layers)
		current = layer->process(current);

	auto it1 = m_outputs.begin();
	auto it2 = current.cbegin();
	while (it1 != m_outputs.end() || it2 != current.cend()) {
		*it1 = *it2;
		it1++; it2++;
	}
}
void mnn::MatrixLayeredBackpropagationNeuralNetwork::add_layer(size_t const& size, bool bias, Value const& minimum_weight_value, Value const& maximum_weight_value) {
	m_layers.push_back(std::make_shared<BackpropagationLayer>(size, m_layers.empty() ? m_inputs.size() : m_layers.back()->size(), bias, minimum_weight_value, maximum_weight_value));
}
void mnn::MatrixLayeredBackpropagationNeuralNetwork::add_layer(size_t const& size, bool bias, std::function<Value(size_t, size_t)> const& weight_function) {
	m_layers.push_back(std::make_shared<BackpropagationLayer>(size, m_layers.empty() ? m_inputs.size() : m_layers.back()->size(), bias, weight_function));
}

void mnn::MatrixLayeredBackpropagationNeuralNetwork::backpropagate(NeuronContainer<Value> const& _outputs) {
	if (_outputs.size() != m_outputs.size())
		throw Exceptions::UnsupportedInputError("Unsupported number of outputs was passed.");
	if (m_layers.empty())
		return;

	mnn::NeuronContainer<mnn::Value> current_gradient(m_outputs.size());
	
	auto layer = m_layers.rbegin();
	auto next = layer;
	while (layer != m_layers.rend()) {
		if (layer == m_layers.rbegin())
			for (size_t i = 0; i < m_outputs.size(); i++)
				current_gradient.at(i) = (_outputs.at(i) - m_outputs.at(i)) * normalization_derivative(m_outputs.at(i));
		else {
			mnn::NeuronContainer<mnn::Value> next_gradient((*layer)->m_weights.front().size());

			for (size_t i = 0; i < (*layer)->m_weights.front().size(); i++) {
				for (size_t j = 0; j < current_gradient.size(); j++)
					next_gradient.at(i) += current_gradient.at(j) * (*next)->m_weights.at(i).at(j);
				next_gradient.at(i) *= normalization_derivative((*layer)->m_value.at(i));
			}
			current_gradient = next_gradient;
		}

		for (size_t i = 0; i < (*layer)->m_deltas.size(); i++)
			for (size_t j = 0; j < (*layer)->m_deltas.front().size(); j++) {
				if (layer == --m_layers.rend())
					(*layer)->m_deltas.at(i).at(j) = m_eta * current_gradient.at(j)
					* (i < (*layer)->m_deltas.size() - 1 ? m_inputs.at(i) : 1.0)
					+ m_alpha * (*layer)->m_deltas.at(i).at(j);
				else {
					auto next = layer;
					(*layer)->m_deltas.at(i).at(j) = m_eta * current_gradient.at(j)
						* (i < (*layer)->m_deltas.size() - 1 ? (*++next)->m_value.at(i) : 1.0)
						+ m_alpha * (*layer)->m_deltas.at(i).at(j);
				}

				(*layer)->m_weights.at(i).at(j) += (*layer)->m_deltas.at(i).at(j);
			}

		next = layer++;
	}
}

#include "mnn/storage/Storage.hpp"
std::ostream& mnn::MatrixLayeredNeuralNetwork::to_stream(std::ostream &output) const {
	output << short(typecodes::matrix_layered_network) << ' '
		<< m_inputs.size() << ' ' << m_outputs.size() << ' ' 
		<< m_layers.size() << '\n';
	for (auto &it : m_layers)
		output << " " << *it << '\n';
	return output;
}
std::ostream& mnn::MatrixLayeredBackpropagationNeuralNetwork::to_stream(std::ostream &output) const {
	output << short(typecodes::matrix_layered_network_backpropagation) << ' '
		<< m_inputs.size() << ' ' << m_outputs.size() << ' '
		<< m_layers.size() << ' '
		<< m_eta << ' ' << m_alpha << '\n';
	for (auto &it : m_layers)
		output << " " << *it << '\n';
	return output;
}
std::istream& mnn::MatrixLayeredNeuralNetwork::from_stream(std::istream &input) {
	size_t size;
	input >> size;
	for (size_t i = 0; i < size; i++)
		m_layers.push_back(Layer::read(input));
	return input;
}
std::istream& mnn::MatrixLayeredBackpropagationNeuralNetwork::from_stream(std::istream &input) {
	size_t size;
	input >> size >> m_eta >> m_alpha;
	for (size_t i = 0; i < size; i++)
		m_layers.push_back(BackpropagationLayer::read(input));
	return input;
}

#include <random>
std::shared_ptr<mnn::MatrixLayeredNeuralNetwork> mnn::MatrixLayeredNeuralNetwork::generate(MatrixLayeredNeuralNetwork const& n1, MatrixLayeredNeuralNetwork const& n2, Value const& ratio) {
	if (n1.inputs().size() != n2.inputs().size() || n1.outputs().size() != n2.outputs().size())
		throw Exceptions::UnsupportedInputError();

	static std::mt19937_64 g(std::random_device{}());
	std::bernoulli_distribution b(ratio);

	auto ret = std::make_shared<MatrixLayeredNeuralNetwork>(n1.inputs().size(), n1.outputs().size());
	while (ret->layers().size() < n1.layers().size() && ret->layers().size() < n2.layers().size())
		ret->layers().push_back(Layer::generate(
			ret->layers().empty() ? ret->inputs().size() : ret->layers().back()->size(),
			*n1.layers().at(ret->layers().size()),
			*n2.layers().at(ret->layers().size())
		));
	while (ret->layers().size() < n1.layers().size())
		ret->layers().push_back(Layer::generate(
			ret->layers().empty() ? ret->inputs().size() : ret->layers().back()->size(),
			*n1.layers().at(ret->layers().size())
		));
	while (ret->layers().size() < n2.layers().size())
		ret->layers().push_back(Layer::generate(
			ret->layers().empty() ? ret->inputs().size() : ret->layers().back()->size(),
			*n2.layers().at(ret->layers().size())
		));

	return ret;
}

void mnn::MatrixLayeredNeuralNetwork::for_each_weight(std::function<void(Value&)> lambda) {
	for (auto &layer : m_layers)
		layer->for_each_weight(lambda);
}
void mnn::MatrixLayeredBackpropagationNeuralNetwork::for_each_weight(std::function<void(Value&)> lambda) {
	for (auto &layer : m_layers)
		layer->for_each_weight(lambda);
}