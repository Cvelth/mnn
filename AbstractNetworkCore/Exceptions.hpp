#pragma once

namespace mnn {
	namespace Exceptions {
		//Given inputs number isn't equal to the expected amount.
		class WrongInputsNumberException {};
		//Given outputs number isn't equal to the expected amount.
		class WrongOutputNumberException {};

		//Requested index do not exist.
		class NonExistingIndexException {};
	}
}