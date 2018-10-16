#pragma once
#include "mnn/interfaces/Types.hpp"
#include <functional>
namespace mnn {
	/*
		Type of selection process:
		- Number - exactly 'm_selection_value' percent of the population survives.
		- Value - only units with maximum_value - (maximum_value - minimum_value) / m_selection_value survive.
	*/
	enum class SelectionType {
		Number,
		Value
	};

	class EvolutionManagerInterface {
	//public:
		//using EvaluationFunction = /?/
	protected:
		std::pair<SelectionType, Value> m_selection_value;
	public:
		EvolutionManagerInterface(SelectionType const& type = SelectionType::Number,
								  Value const& selection_value = Value(0.5))
			: m_selection_value(type, selection_value) {}

		inline std::pair<SelectionType, Value>& selection_parameters(SelectionType const& type = SelectionType::Number,
															  Value const& selection_value = Value(0.5)) {
			return m_selection_value = std::pair(type, selection_value);
		}
		inline std::pair<SelectionType, Value> const& selection_parameters() const {
			return m_selection_value;
		}

		virtual void fill(bool base_on_existing = true) = 0;
		virtual void select() = 0;
		virtual void mutate(float unit_mutation_chance, float weight_mutation_chance) = 0;

		inline void next_generation(float unit_mutation_chance, float weight_mutation_chance) {
			fill(true);
			select();
			mutate(unit_mutation_chance, weight_mutation_chance);
		}
	};
}