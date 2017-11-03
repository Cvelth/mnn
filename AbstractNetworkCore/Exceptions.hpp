#pragma once
#include <exception>
namespace mnn {
	namespace Exceptions {
		//Given inputs number isn't equal to the expected amount.
		class WrongInputsNumberException : std::exception {};
		//Given outputs number isn't equal to the expected amount.
		class WrongOutputNumberException : std::exception {};

		//Requested index do not exist.
		class NonExistingIndexException : std::exception {};
	}
}