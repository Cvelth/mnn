#pragma once
#include <functional>
#include <memory>
#include "mnn/interfaces/Types.hpp"
#include "mnn/interfaces/EvolutionManagerInterface.hpp"
namespace mnn {
	class NeuralNetworkInterface;

	class NeuralNetworkEvolutionManager : public EvolutionManagerInterface<
		std::function<Value(std::shared_ptr<NeuralNetworkInterface>)>
	> {
	protected:
		NetworkContainer<std::shared_ptr<NeuralNetworkInterface>> m_population;
	public:
		using EvolutionManagerInterface::EvolutionManagerInterface;

		virtual void fill(bool base_on_existing = true) override;
		virtual void select() override;
		virtual void mutate(Value unit_mutation_chance, Value weight_mutation_chance) override;

		auto const& operator*() const { return m_population; }
		auto& operator*() { return m_population; }
		auto const* operator->() const { return &m_population; }
		auto* operator->() { return &m_population; }
	};
}