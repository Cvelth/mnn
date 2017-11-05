#pragma once
using Type = float;

#include <vector>
template <typename InnerType>
using NeuronContainer = std::vector<InnerType>;
template <typename InnerType>
using LinkContainer = std::vector<InnerType>;
template <typename InnerType>
using LayerContainer = std::vector<InnerType>;

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
	GenerateNewException(NonExistingNetworkUsed);
}