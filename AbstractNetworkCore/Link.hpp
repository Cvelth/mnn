#pragma once

namespace MNN {
	template <typename T>
	class AbstractNeuron;

	template <typename T>
	struct Link {
		AbstractNeuron<T>* unit;
		T weight;

		Link(AbstractNeuron<T>* unit, T weight) : unit(unit), weight(weight) {}
	};
}