#pragma once
using Type = float;

#include <vector>
#define NeuronContainer std::vector
#define LinkContainer std::vector
#define LayerContainer std::vector

#include <exception>
#define GenerateNewException(name) 		   \
namespace Exceptions {					   \
	class name : public std::exception {   \
	public: using exception::exception;	   \
	};									   \
}
namespace mnn {
	GenerateNewException(IncorrectDataAmountException);
	GenerateNewException(NonExistingIndexException);
}