#pragma once
#include "Shared.hpp"
#include "AbstractEvolutionManager.hpp"
namespace mnn {
	GenerateNewException(UnofficientAmountOfSurvivors)
	class AbstractNetwork;
	class NetworkGenerationEvolutionManager : public AbstractNetworkGenerationEvolutionManager {
	protected:
		NetworkContainer<std::pair<Type, AbstractNetwork*>> m_networks;
	public:
		using AbstractNetworkGenerationEvolutionManager::AbstractNetworkGenerationEvolutionManager;
		~NetworkGenerationEvolutionManager();
		virtual void newPopulation() override;
		virtual void testPopulation() override;
		virtual void selectionStep() override;
		virtual void recreatePopulation(bool baseOnSurvivors = true) override;
	};
}