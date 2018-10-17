#include "NeuralNetworkEvolutionManager.hpp"
#include <random>
void mnn::NeuralNetworkEvolutionManager::fill(bool base_on_existing) {
	if (m_population.size() > m_population_size)
		throw Exceptions::BrokenStateError();
	m_population.reserve(m_population_size);
	if (base_on_existing && m_population.size() > 2) {
		static std::mt19937_64 g(std::random_device{}());
		std::uniform_int_distribution<size_t> d(0, m_population.size() - 1);
		while (m_population.size() < m_population_size)
			m_population.push_back(m_breeding_function(m_population.at(d(g)), m_population.at(d(g))));
	} else 
		while (m_population.size() < m_population_size)
			m_population.push_back(m_generation_function());
}
void mnn::NeuralNetworkEvolutionManager::select() {

}
void mnn::NeuralNetworkEvolutionManager::mutate(Value unit_mutation_chance, Value weight_mutation_chance) {

}