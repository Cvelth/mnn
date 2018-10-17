#pragma once
#include <functional>
#include <memory>
#include "mnn/interfaces/Types.hpp"
#include "mnn/interfaces/EvolutionManagerInterface.hpp"

#include "mnn/exceptions.hpp"
DefineNewMNNException(BrokenStateError);

namespace mnn {
	class NeuralNetworkInterface;

	class NeuralNetworkEvolutionManager : public EvolutionManagerInterface<
		std::function<Value(std::shared_ptr<NeuralNetworkInterface>)>
	> {
	public:
		using EvaluationFunction = std::function<Value(std::shared_ptr<NeuralNetworkInterface>)>;
		using BreedingFunction = std::function<std::shared_ptr<NeuralNetworkInterface>(
			std::shared_ptr<NeuralNetworkInterface>,
			std::shared_ptr<NeuralNetworkInterface>
		)>;
		using GenerationFunction = std::function<std::shared_ptr<NeuralNetworkInterface>()>;
	protected:
		NetworkContainer<std::pair<std::shared_ptr<NeuralNetworkInterface>, Value>> m_population;

		GenerationFunction m_generation_function;
		BreedingFunction m_breeding_function;
	public:
		NeuralNetworkEvolutionManager(size_t const& population_size,
									  size_t const& input_number,
									  size_t const& output_number,
									  EvaluationFunction const& ev_function,
									  GenerationFunction const& gn_function,
									  BreedingFunction const& br_function,
									  SelectionType const& type = SelectionType::Number,
									  Value const& selection_value = Value(0.5))
					: EvolutionManagerInterface(population_size, input_number, output_number,
										ev_function, type, selection_value),
					m_generation_function(gn_function), m_breeding_function(br_function) {
		
			fill(false);
		}

		inline void generation_function(GenerationFunction const& function) { m_generation_function = function; }
		inline GenerationFunction const& generation_function() const { return m_generation_function; }
		inline void breeding_function(BreedingFunction const& function) { m_breeding_function = function; }
		inline BreedingFunction const& breeding_function() const { return m_breeding_function; }

		virtual void fill(bool base_on_existing = true) override;
		virtual void select() override;
		virtual void mutate(Value const& unit_mutation_chance, Value const& weight_mutation_chance) override;

		auto const& operator*() const { return m_population; }
		auto& operator*() { return m_population; }
		auto const* operator->() const { return &m_population; }
		auto* operator->() { return &m_population; }
	};
}