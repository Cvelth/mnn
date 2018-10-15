#include "NeuralNetwork.hpp"
#include "mnn/exceptions.hpp"
DefineNewMNNException(UnimplementedFeature);

#include "neuron.hpp"
mnn::ExplicitlyLinkedNeuralNetwork::ExplicitlyLinkedNeuralNetwork(size_t input_number, size_t output_number)
	: NeuralNetworkInterface(input_number, output_number) {
	for (size_t i = 0; i < input_number; i++)
		m_input_neurons.push_back(std::make_shared<Neuron>());
	for (size_t i = 0; i < output_number; i++)
		m_output_neurons.push_back(std::make_shared<Neuron>());
}
mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::ExplicitlyLinkedBackpropagationNeuralNetwork(size_t input_number, size_t output_number)
	: BackpropagationNeuralNetworkInterface(input_number, output_number) {
	for (size_t i = 0; i < input_number; i++)
		m_input_neurons.push_back(std::make_shared<BackpropagationNeuron>());
	for (size_t i = 0; i < output_number; i++)
		m_output_neurons.push_back(std::make_shared<BackpropagationNeuron>());
}

void mnn::ExplicitlyLinkedNeuralNetwork::process() {
	auto it1 = m_inputs.cbegin();
	auto it2 = m_input_neurons.begin();
	while (it1 != m_inputs.cend() || it2 != m_input_neurons.end()) {
		**it2 = *it1;
		it1++; it2++;
	}

	auto it3 = m_outputs.begin();
	auto it4 = m_output_neurons.begin();
	while (it3 != m_outputs.end() || it4 != m_output_neurons.end()) {
		*it3 = (*it4)->value(true);
		it3++; it4++;
	}
}
void mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::process() {
	auto it1 = m_inputs.cbegin();
	auto it2 = m_input_neurons.begin();
	while (it1 != m_inputs.cend() || it2 != m_input_neurons.end()) {
		**it2 = *it1;
		it1++; it2++;
	}

	auto it3 = m_outputs.begin();
	auto it4 = m_output_neurons.begin();
	while (it3 != m_outputs.end() || it4 != m_output_neurons.end()) {
		*it3 = (*it4)->value(true);
		it3++; it4++;
	}
}

void mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::backpropagate(NeuronContainer<Value> const & _outputs) {
	//TO DO.
}

#include "mnn/storage/Storage.hpp"
std::ostream& mnn::ExplicitlyLinkedNeuralNetwork::to_stream(std::ostream &output) const {
	output << short(typecodes::explicitly_linked_network) << ' ' 
		<< m_inputs.size() << ' ' << m_outputs.size() << '\n';
	for (auto &it : m_input_neurons)
		output << *it;
	for (auto &it : m_output_neurons)
		output << *it;
	for (auto &it : m_hidden_neurons)
		output << *it;
	return output;
}
std::ostream& mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::to_stream(std::ostream &output) const {
	output << short(typecodes::explicitly_linked_network) << ' '
		<< m_inputs.size() << ' ' << m_outputs.size() << '\n';
	for (auto &it : m_input_neurons)
		output << *it;
	for (auto &it : m_output_neurons)
		output << *it;
	for (auto &it : m_hidden_neurons)
		output << *it;
	return output;
}

#include <map>
std::istream& mnn::ExplicitlyLinkedNeuralNetwork::from_stream(std::istream &input) {
	std::map<size_t, std::shared_ptr<NeuronInterface>> neurons;
	std::map<size_t, std::vector<std::pair<size_t, Value>>> links;

	short type;
	size_t id, current_id = -1;
	Value value;

	auto next_neuron = [&]() -> auto {
		static size_t i = -1;
		if (++i < m_input_neurons.size())
			return m_input_neurons.at(i);
		if (i < m_output_neurons.size() + m_input_neurons.size())
			return m_output_neurons.at(i - m_input_neurons.size());
		m_hidden_neurons.push_back(std::make_shared<Neuron>());
		return m_hidden_neurons.back();
	};

	while (input >> type) {
		switch (typecodes(type)) {
			case typecodes::neuron:
			case typecodes::neuron_backpropagation:
				input >> current_id;
				neurons.insert(std::pair(current_id, next_neuron()));
				links.insert(std::pair(current_id, std::vector<std::pair<size_t, Value>>{}));
				break;

			case typecodes::link:
			case typecodes::link_backpropagation:
				if (current_id == -1)
					throw Exceptions::UnsupportedFileError();
				input >> id >> value;
				links.at(current_id).push_back(std::pair(id, value));
				break;

			default:
				throw Exceptions::UnsupportedFileError();
		}
	}

	for (auto &neuron : neurons)
		for (auto &link : links.at(neuron.first))
			neuron.second->link(neurons.at(link.first),
								link.second);
	return input;
}

#include <map>
std::istream& mnn::ExplicitlyLinkedBackpropagationNeuralNetwork::from_stream(std::istream &input) {
	std::map<size_t, std::shared_ptr<NeuronInterface>> neurons;
	std::map<size_t, std::vector<std::tuple<size_t, Value, Value>>> links;

	short type;
	size_t id, current_id = -1;
	Value weight, delta;

	auto next_neuron = [&]() -> auto {
		static size_t i = -1;
		if (++i < m_input_neurons.size())
			return m_input_neurons.at(i);
		if (i < m_output_neurons.size() + m_input_neurons.size())
			return m_output_neurons.at(i - m_input_neurons.size());
		m_hidden_neurons.push_back(std::make_shared<BackpropagationNeuron>());
		return m_hidden_neurons.back();
	};

	while (input >> type) {
		switch (typecodes(type)) {
			case typecodes::neuron:
			case typecodes::neuron_backpropagation:
				input >> current_id;
				neurons.insert(std::pair(current_id, next_neuron()));
				links.insert(std::pair(current_id, std::vector<std::tuple<size_t, Value, Value>>{}));
				break;

			case typecodes::link:
			case typecodes::link_backpropagation:
				if (current_id == -1)
					throw Exceptions::UnsupportedFileError();
				input >> id >> weight >> delta;
				links.at(current_id).push_back(std::tuple(id, weight, delta));
				break;

			default:
				throw Exceptions::UnsupportedFileError();
		}
	}

	for (auto &neuron : neurons)
		for (auto &link : links.at(neuron.first))
			std::dynamic_pointer_cast<BackpropagationNeuron>(neuron.second)->link(
				neurons.at(std::get<0>(link)),
				std::get<1>(link),
				std::get<2>(link)
			);
	return input;
}