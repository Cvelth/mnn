#pragma once
#include "Shared.hpp"
#include <functional>
namespace mnn {
	using EvaluationFunction = std::function<Type(std::function<NeuronContainer<Type>(NeuronContainer<Type>)>)>;
	class AbstractEvolutionManager {
	protected:
		EvaluationFunction m_evaluate;
	public:
		explicit AbstractEvolutionManager() {}
		explicit AbstractEvolutionManager(EvaluationFunction f) : m_evaluate(f) {}
		void changeEvaluationFunction(EvaluationFunction f) { m_evaluate = f; }
		virtual void nextGeneration() =0;
	};
	class AbstractGenerationEvolutionManager : public AbstractEvolutionManager {
	private:
		size_t m_units;
	public:
		explicit AbstractGenerationEvolutionManager(size_t units)
			: AbstractEvolutionManager(), m_units(units) {}
		explicit AbstractGenerationEvolutionManager(size_t units, EvaluationFunction f)
			: AbstractEvolutionManager(f), m_units(units) {}
	};
}