#pragma once
#include <exception>
namespace mnn {
	namespace Exceptions {
		class IncorrectDataException : std::exception {};
		class NonExistingIndexException : std::exception {};
	}
}