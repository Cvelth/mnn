#pragma once
#define Type float

#include <vector>
#define NeuronContainer std::vector
#define LinkContainer std::vector
#define LayerContainer std::vector

#include <exception>
namespace mnn {
	namespace Exceptions {
		class IncorrectDataAmountException : std::exception {};
		class NonExistingIndexException : std::exception {};
	}
}