#pragma once
#include "Shared.hpp"
namespace mnn {
	class AbstractNeuron;
	struct Link {
		AbstractNeuron* unit;
		Type weight;
		Type delta;
		Link(AbstractNeuron* unit, Type const& weight) : unit(unit), weight(weight), delta(0.f) {}
		inline void step() { weight += delta; }
	};
}