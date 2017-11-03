#pragma once
#define Type float

#include <vector>
#define NeuronContainer std::vector
#define LinkContainer std::vector

#include <exception>
namespace mnn {
	namespace Exceptions {
		class IncorrectDataException : std::exception {};
		class NonExistingIndexException : std::exception {};
	}
}