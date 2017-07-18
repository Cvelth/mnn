#include "Layer.hpp"

mnn::AbstractDataContainerLayer::~AbstractDataContainerLayer() {
	for (auto neuron : m_neurons)
		delete neuron;
}
