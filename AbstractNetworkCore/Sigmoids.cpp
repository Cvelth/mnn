#include "Sigmoids.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

float MNN::erf_sigmoid(const float & x) {
	return erff(x);
}
float MNN::tanh_sigmoid(const float & x) {
	return tanh(x);
}
float MNN::root_sigmoid(const float & x) {
	return x / sqrtf(1 + x * x);
}
float MNN::atan_sigmoid(const float & x) {
	return 2.f / M_PI * atanf(x * M_PI / 2.f);
}
float MNN::erf_sigmoid_derivative(const float & x) {
	return 2.f * expf(-x * x) / sqrtf(M_PI);
}
float MNN::tanh_sigmoid_derivative(const float & x) {
	float t = tanhf(x);
	return 1 - t * t;
}
float MNN::tanh_sigmoid_derivative_approximated(const float & x) {
	return 1 - x * x;
}
float MNN::root_sigmoid_derivative(const float & x) {
	return -x * x / powf(1 + x * x, 3.f / 2.f) + 1.f / sqrtf(1 + x * x);
}
float MNN::atan_sigmoid_derivative(const float & x) {
	return 1.f / (1 + M_PI * M_PI * x * x / 4.f);
}