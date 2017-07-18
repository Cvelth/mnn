#pragma once

namespace mnn {
	class AbstractNeuron;

	struct Link {
		AbstractNeuron* unit;
		float weight;
		float delta;

		Link(AbstractNeuron* unit, float weight) : unit(unit), weight(weight), delta(0.f) {}
		inline void step() {
			weight += delta;
		}
	};
}