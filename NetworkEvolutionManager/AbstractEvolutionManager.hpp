#pragma once
#include "Shared.hpp"
#include <functional>
namespace mnn {
	/*
	Determines type of selection process.

	* When the Number is set, exactly X persent of the population will survive.
	* When the Value is set, to survive Unit needs to have more than
		maximum_value - (maximum_value - minimum_value) / X, as its score.

	Default value: Number.
	*/
	enum class SelectionType {
		Number,
		Value
	};
	GenerateNewException(UnlogicalParameterWasPassed)
	using EvaluationFunction = std::function<Type(std::function<NeuronContainer<Type>(NeuronContainer<Type>)>)>;
	class AbstractEvolutionManager {
	protected:
		EvaluationFunction m_evaluate;
		SelectionType m_selection_type;
		float m_selection_persent;
	public:
		explicit AbstractEvolutionManager() : m_selection_type(SelectionType::Number), m_selection_persent(0.5f) {}
		explicit AbstractEvolutionManager(EvaluationFunction f) : m_evaluate(f), 
			m_selection_type(SelectionType::Number), m_selection_persent(0.5f) {}
		void changeEvaluationFunction(EvaluationFunction f) { m_evaluate = f; }
		void changeSelectionParameters(float selection_persent = 0.5f, SelectionType selection_type = SelectionType::Number) {
			m_selection_type = selection_type;
			m_selection_persent = selection_persent;
		}
		void changeSelectionParameters(SelectionType selection_type, float selection_persent = 0.5f) {
			changeSelectionParameters(selection_persent, selection_type);
		}
		virtual void newPopulation() =0;
		virtual Type testPopulation(bool sort = true) =0;
		virtual void sortPopulation() =0;
		virtual void populationSelection() =0;
		virtual void recreatePopulation(bool baseOnSurvivors = true) =0;
		virtual void mutatePopulation(float unit_mutation_chance, float weight_mutation_chance) =0;
	};
	class AbstractGenerationEvolutionManager : public AbstractEvolutionManager {
	protected:
		size_t m_units;
	public:
		explicit AbstractGenerationEvolutionManager(size_t units)
			: AbstractEvolutionManager(), m_units(units) {}
		explicit AbstractGenerationEvolutionManager(size_t units, EvaluationFunction f)
			: AbstractEvolutionManager(f), m_units(units) {}
	};
	class AbstractNetworkGenerationEvolutionManager : public AbstractGenerationEvolutionManager {
	protected:
		size_t m_input_neurons;
		size_t m_output_neurons;
		size_t m_hidden_neurons;
		size_t m_layer_divisor;
	protected:
		void check_divisor() {
			if (m_layer_divisor == 0)
				throw Exceptions::UnlogicalParameterWasPassed("The Divisor mustn't be equal to zero.");
			if (m_hidden_neurons % m_layer_divisor != 0)
				throw Exceptions::UnlogicalParameterWasPassed("Number of hidden layer must be a multicant of the Divisor.");
		}
	public:
		explicit AbstractNetworkGenerationEvolutionManager(size_t units, size_t input_neurons, size_t output_neurons,
														   size_t hidden_neurons = 0u, size_t layer_divisor = 1u)
			: AbstractGenerationEvolutionManager(units), m_input_neurons(input_neurons), m_output_neurons(output_neurons),
			m_hidden_neurons(hidden_neurons), m_layer_divisor(layer_divisor) { check_divisor(); }

		explicit AbstractNetworkGenerationEvolutionManager(size_t units, EvaluationFunction f,
														   size_t input_neurons, size_t output_neurons,
														   size_t hidden_neurons = 0u, size_t layer_divisor = 1u)
			: AbstractGenerationEvolutionManager(units, f), m_input_neurons(input_neurons), m_output_neurons(output_neurons),
			m_hidden_neurons(hidden_neurons), m_layer_divisor(layer_divisor) { check_divisor();	}
	};
}