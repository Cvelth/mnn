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

float MNN::abs_sigmoid(const float & x) {
	return x / (1 + fabs(x));
}
