#pragma once

namespace mnn {
	float erf_sigmoid(const float& x);
	float tanh_sigmoid(const float& x);
	float root_sigmoid(const float& x);
	float atan_sigmoid(const float& x);
	float erf_sigmoid_derivative(const float& x);
	float tanh_sigmoid_derivative(const float& x);
	float tanh_sigmoid_derivative_approximated(const float& x);
	float root_sigmoid_derivative(const float& x);
	float atan_sigmoid_derivative(const float& x);
}