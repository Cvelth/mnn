#pragma once
#include "Shared.hpp"
namespace mnn {
	Type erf_sigmoid(Type x);
	Type tanh_sigmoid(Type x);
	Type root_sigmoid(Type x);
	Type atan_sigmoid(Type x);
	Type erf_sigmoid_derivative(Type x);
	Type tanh_sigmoid_derivative(Type x);
	Type tanh_sigmoid_derivative_approximated(Type x);
	Type root_sigmoid_derivative(Type x);
	Type atan_sigmoid_derivative(Type x);
}