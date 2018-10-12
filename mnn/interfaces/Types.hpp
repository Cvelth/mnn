#pragma once
namespace mnn {
	using Value = double;
}

#include <vector>
namespace mnn {
	template <typename Type>
	using NeuronContainer = std::vector<Type>;
	template <typename Type>
	using LinkContainer = std::vector<Type>;
	template <typename Type>
	using LayerContainer = std::vector<Type>;
}

namespace mnn {
	static Value normalize(Value const& value);
	static Value normalization_derivative(Value const& value);
}