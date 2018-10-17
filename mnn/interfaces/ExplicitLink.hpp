#pragma once
#include "mnn/interfaces/Types.hpp"
namespace mnn {
	class NeuronInterface;
	struct ExplicitLink {
		std::shared_ptr<NeuronInterface> unit;
		Value weight;
		ExplicitLink(std::shared_ptr<NeuronInterface> unit, Value const& weight) : unit(unit), weight(weight) {}
	};
	struct ExplicitBackpropagationLink : public ExplicitLink {
		Value delta;
		ExplicitBackpropagationLink(std::shared_ptr<NeuronInterface> unit, Value const& weight, Value const& delta = 0.f)
			: ExplicitLink(unit, weight), delta(delta) {}
		ExplicitBackpropagationLink(ExplicitLink const& other, Value const& delta = 0.f)
			: ExplicitLink(other.unit, other.weight), delta(delta) {}
		inline void step() { weight += delta; }
		inline void operator()() { step(); }
	};
}