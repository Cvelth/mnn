#define _USE_MATH_DEFINES
#include "Sigmoids.hpp"
Type mnn::erf_sigmoid(Type x) {
	return erff(x);
}
Type mnn::tanh_sigmoid(Type x) {
	return tanh(x);
}
Type mnn::root_sigmoid(Type x) {
	return x / sqrtf(1 + x * x);
}
Type mnn::atan_sigmoid(Type x) {
	return 2.f / M_PI * atanf(x * M_PI / 2.f);
}
Type mnn::erf_sigmoid_derivative(Type x) {
	return 2.f * expf(-x * x) / sqrtf(M_PI);
}
Type mnn::tanh_sigmoid_derivative(Type x) {
	Type t = tanhf(x);
	return 1 - t * t;
}
Type mnn::tanh_sigmoid_derivative_approximated(Type x) {
	return 1 - x * x;
}
Type mnn::root_sigmoid_derivative(Type x) {
	return -x * x / powf(1 + x * x, 3.f / 2.f) + 1.f / sqrtf(1 + x * x);
}
Type mnn::atan_sigmoid_derivative(Type x) {
	return 1.f / (1 + M_PI * M_PI * x * x / 4.f);
}