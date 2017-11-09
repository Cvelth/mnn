#pragma once
#include <functional> 
#include "Shared.hpp"
#include "AbstractEvolutionManager.hpp"
namespace mnn {
	GenerateNewException(UnofficientAmountOfSurvivors)
	class AbstractNetwork;
	class NetworkGenerationEvolutionManager : public AbstractNetworkGenerationEvolutionManager {
	protected:
		NetworkContainer<std::pair<Type, AbstractNetwork*>> m_networks;
	public:
		virtual void mutate(AbstractNetwork *unit, float mutation_chance);
	public:
		using AbstractNetworkGenerationEvolutionManager::AbstractNetworkGenerationEvolutionManager;
		~NetworkGenerationEvolutionManager();
		virtual void newPopulation() override;
		virtual void testPopulation(bool sort = true) override;
		virtual void sortPopulation() override;
		virtual void populationSelection() override;
		virtual void recreatePopulation(bool baseOnSurvivors = true) override;
		virtual void mutatePopulation(float unit_mutation_chance,
									  float weight_mutation_chance) override;

		virtual void for_each(std::function<void(std::pair<Type, AbstractNetwork*>)> lambda, bool firstToLast = true);
		virtual void for_each(std::function<void(std::pair<Type, AbstractNetwork*>)> lambda, bool firstToLast = true) const;
		NetworkContainer<std::pair<Type, AbstractNetwork*>> const& operator*() const;
		NetworkContainer<std::pair<Type, AbstractNetwork*>>& operator*();
	};
}