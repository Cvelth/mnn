#include "NetworkGenerationEvolutionManager.hpp"
#include "AbstractLayerNetwork.hpp"
#include "Automatization.hpp"
mnn::NetworkGenerationEvolutionManager::~NetworkGenerationEvolutionManager() {
	for (auto &it : m_networks)
		delete it.second;
}
void mnn::NetworkGenerationEvolutionManager::newPopulation() {
	if (m_networks.size() != 0u) 
		while (m_networks.size()) {
			delete m_networks.back().second;
			m_networks.pop_back();
	}
	recreatePopulation(false);
}
void mnn::NetworkGenerationEvolutionManager::testPopulation(bool sort) {
	for (auto &it : m_networks) {
		it.first = m_evaluate([&it](NeuronContainer<Type> inputs) -> NeuronContainer<Type> {
			it.second->calculate(inputs);
			return it.second->getOutputs();
		});
	}
	if (sort) sortPopulation();
}
#include <algorithm>
void mnn::NetworkGenerationEvolutionManager::sortPopulation() {
	std::sort(m_networks.begin(), m_networks.end(), [](auto l, auto r) { return l.first > r.first; });
}
void mnn::NetworkGenerationEvolutionManager::populationSelection() {
	if (!std::is_sorted(m_networks.begin(), m_networks.end(), [](auto l, auto r) { return l.first > r.first; }))
		sortPopulation();
	switch (m_selection_type) {
		Type mid_value;
		case SelectionType::Number:
			mid_value = m_units * m_selection_persent;
			while (m_networks.size() > mid_value) {
				delete m_networks.back().second;
				m_networks.pop_back();
			}
			break;
		case SelectionType::Value:
			mid_value = m_networks.front().first
				- (m_networks.front().first - m_networks.back().first) * m_selection_persent;

			while (m_networks.back().first < mid_value) {
				delete m_networks.back().second;
				m_networks.pop_back();
			}
			break;
	}
}
#include <random>
void mnn::NetworkGenerationEvolutionManager::recreatePopulation(bool baseOnSurvivors) {
	m_networks.reserve(m_units);
	if (baseOnSurvivors) {
		size_t survivors = m_networks.size();
		if (survivors < 3) throw Exceptions::UnofficientAmountOfSurvivors();
		std::mt19937_64 g((std::random_device())());
		std::uniform_int_distribution<size_t> d(0, survivors - 1);
		while (m_networks.size() < m_units) 
			m_networks.push_back(std::make_pair(0.f, generateTypicalLayerNeuralNetwork(dynamic_cast<mnn::AbstractLayerNetwork*>(m_networks[d(g)].second),
																					   dynamic_cast<mnn::AbstractLayerNetwork*>(m_networks[d(g)].second))));
	} else {
		for (size_t i = m_networks.size(); i < m_units; i++)
			m_networks.push_back(std::make_pair(0.f, generateTypicalLayerNeuralNetwork(m_input_neurons, m_output_neurons,
																					   m_layer_divisor, m_hidden_neurons / m_layer_divisor,
																					   ConnectionPattern::EachFromPreviousLayerWithBias, random_weights)));
	}
}
void mnn::NetworkGenerationEvolutionManager::mutatePopulation(float unit_mutation_chance, float weight_mutation_chance) {
	std::mt19937_64 g((std::random_device())());
	std::uniform_real_distribution<float> d(0.f, 1.f);
	if (unit_mutation_chance > 1.f || unit_mutation_chance < 0.f || weight_mutation_chance > 1.f || weight_mutation_chance < 1.f)
		throw Exceptions::UnlogicalParameterWasPassed();
	for (auto& it : m_networks)
		if (d(g) < unit_mutation_chance)
			mutate(it.second, weight_mutation_chance);
}
#include "AbstractNeuron.hpp"
#include "Link.hpp"
void mnn::NetworkGenerationEvolutionManager::mutate(AbstractNetwork *abstract_unit, float mutation_chance) {
	std::mt19937_64 g((std::random_device())());
	std::uniform_real_distribution<float> d(0.f, 1.f);
	std::uniform_real_distribution<float> w(-1.f, +1.f);
	auto unit = dynamic_cast<AbstractLayerNetwork*>(abstract_unit);
	unit->for_each_neuron([&](AbstractNeuron& n) {
		n.for_each_link([&](Link& l) {
			if (d(g) < mutation_chance)
				l.weight = w(g);
		});
	});
}