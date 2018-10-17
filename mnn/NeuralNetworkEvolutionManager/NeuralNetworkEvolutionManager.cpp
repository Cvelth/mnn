#include "NeuralNetworkEvolutionManager.hpp"
#include <random>
#include <algorithm>
void mnn::NeuralNetworkEvolutionManager::fill(bool base_on_existing) {
	if (m_population.size() > m_population_size)
		throw Exceptions::BrokenStateError();
	m_population.reserve(m_population_size);
	if (base_on_existing && m_population.size() > 2) {
		static std::mt19937_64 g(std::random_device{}());
		std::uniform_int_distribution<size_t> d(0, m_population.size() - 1);
		while (m_population.size() < m_population_size)
			m_population.push_back(
				std::pair(m_breeding_function(m_population.at(d(g)).first, 
											  m_population.at(d(g)).first), 
						  0.0)
			);
	} else 
		while (m_population.size() < m_population_size)
			m_population.push_back(std::pair(m_generation_function(), 0.0));
}
void mnn::NeuralNetworkEvolutionManager::select() {
	for (auto &it : m_population)
		it.second = m_evaluation_function(it.first);
	std::sort(m_population.begin(), m_population.end(), [](auto a, auto b) -> bool {
		return a.second > b.second;
	});

	switch (m_selection_value.first) {
		case SelectionType::Number:			
			if (auto target_size = size_t(m_selection_value.second * m_population_size); m_population.size() > target_size)
				m_population.resize(target_size);
			break;
		case SelectionType::Value:
			if (!m_population.empty()) {
				auto condition = m_population.front().second - 
					(m_population.front().second - m_population.back().second) / m_selection_value.second;
				m_population.erase(std::remove_if(m_population.begin(), m_population.end(), [&condition](auto a) -> bool {
					return a.second < condition;
				}), m_population.end());
			}
			break;
		default:
			throw Exceptions::BrokenStateError();
	}
}
void mnn::NeuralNetworkEvolutionManager::mutate(Value const& unit_mutation_chance, Value const& weight_mutation_chance) {

}