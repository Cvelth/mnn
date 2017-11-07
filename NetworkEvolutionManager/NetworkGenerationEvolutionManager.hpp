#pragma once
#include "Shared.hpp"
#include "AbstractEvolutionManager.hpp"
namespace mnn {
	class AbstractNetwork;
	class NetworkGenerationEvolutionManager : public AbstractNetworkGenerationEvolutionManager {
		NetworkContainer<std::pair<Type, AbstractNetwork*>> m_networks;
	public:
		using AbstractNetworkGenerationEvolutionManager::AbstractNetworkGenerationEvolutionManager;
		~NetworkGenerationEvolutionManager();
		virtual void newPopulation() override;
		virtual void testPopulation() override;
		virtual void selectionStep() override;
		virtual void recreatePopulation() override;
	};
}