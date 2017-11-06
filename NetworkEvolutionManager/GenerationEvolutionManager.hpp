#pragma once
#include "Shared.hpp"
#include "AbstractEvolutionManager.hpp"
namespace mnn {
	class AbstractNetwork;
	class GenerationEvolutionManager : public AbstractGenerationEvolutionManager {
		NetworkContainer<AbstractNetwork*> m_networks;
	public:
		explicit GenerationEvolutionManager(size_t units)
			: AbstractGenerationEvolutionManager(units), m_networks(0) {}
		explicit GenerationEvolutionManager(size_t units, EvaluationFunction f)
			: AbstractGenerationEvolutionManager(units, f), m_networks(0) {}
		virtual void nextGeneration() override;
	};
}