#pragma once

namespace MNN {
	class AbstractNeuron;

	struct Link {
		AbstractNeuron* unit;
		float weight;

		Link(AbstractNeuron* unit, float weight) : unit(unit), weight(weight) {}
	};
}