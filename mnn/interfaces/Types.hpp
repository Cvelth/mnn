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
}