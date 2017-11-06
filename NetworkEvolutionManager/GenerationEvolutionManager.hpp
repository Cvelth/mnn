#pragma once
#include "Shared.hpp"
#include "AbstractEvolutionManager.hpp"
namespace mnn {
	class AbstractNetwork;
	class GenerationEvolutionManager : public AbstractGenerationEvolutionManager {
		NetworkContainer<AbstractNetwork*> m_networks;
	public:
		explicit GenerationEvolutionManager(size_t units)
			: AbstractGenerationEvolutionManager(units) {}
		explicit GenerationEvolutionManager(size_t units, EvaluationFunction f)
			: AbstractGenerationEvolutionManager(units, f) {}
		explicit GenerationEvolutionManager(NetworkContainer<AbstractNetwork*> const& m_networks)
			: AbstractGenerationEvolutionManager(m_networks.size()), m_networks(m_networks) {}
		explicit GenerationEvolutionManager(NetworkContainer<AbstractNetwork*> const& m_networks, EvaluationFunction f)
			: AbstractGenerationEvolutionManager(m_networks.size(), f), m_networks(m_networks) {}
	};
}