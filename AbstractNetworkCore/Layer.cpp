#include "Layer.hpp"

MNN::AbstractDataContainerLayer::~AbstractDataContainerLayer() {
	for (auto neuron : m_neurons)
		delete neuron;
}
