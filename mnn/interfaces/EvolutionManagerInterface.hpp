#pragma once
#include "mnn/interfaces/Types.hpp"
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

	template <typename EvaluationFunction>
	class EvolutionManagerInterface {
	protected:
		EvaluationFunction m_evaluation_function;
		std::pair<SelectionType, Value> m_selection_value;
		size_t m_population_size;
		const size_t m_input_number, m_output_number;
	public:
		EvolutionManagerInterface(size_t const& population_size,
								  size_t const& input_number,
								  size_t const& output_number, 
								  EvaluationFunction const& function,
								  SelectionType const& type = SelectionType::Number,
								  Value const& selection_value = Value(0.5))
			: m_population_size(population_size), m_selection_value(type, selection_value),
			m_input_number(input_number), m_output_number(output_number),
			m_evaluation_function(function) {}

		inline void selection_parameters(Value const& selection_value,
										 SelectionType const& type = SelectionType::Number) {
			m_selection_value = std::pair(type, selection_value);
		}
		inline void selection_parameters(SelectionType const& type,
										 Value const& selection_value  = Value(0.5)) {
			m_selection_value = std::pair(type, selection_value);
		}
		inline std::pair<SelectionType, Value> const& selection_parameters() const {
			return m_selection_value;
		}

		inline void population_size(size_t const& size) { m_population_size = size; }
		inline size_t const& population_size() const { return m_population_size; }

		inline size_t const& input_number() const { return m_input_number; }
		inline size_t const& output_number() const { return m_output_number; }

		inline void evaluation_function(EvaluationFunction const& function) { m_evaluation_function = function; }
		inline EvaluationFunction const& evaluation_function() const { return m_evaluation_function; }

		virtual void fill(bool base_on_existing = true) = 0;
		virtual void select() = 0;
		virtual void mutate(Value unit_mutation_chance, Value weight_mutation_chance) = 0;

		inline void next_generation(Value unit_mutation_chance, Value weight_mutation_chance) {
			fill(true);
			select();
			mutate(unit_mutation_chance, weight_mutation_chance);
		}
	};
}