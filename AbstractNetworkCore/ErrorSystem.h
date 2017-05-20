#pragma once
#include "NetworkContainer.hpp"

namespace MNN {
	class AbstractLayerNetwork;
}

namespace MNN {
	class AbstractErrorSystem {
	protected:
		AbstractLayerNetwork* m_network;
	public:
		AbstractErrorSystem(AbstractLayerNetwork* network) : m_network(network) {}
		virtual float calculateNetworkError(const NetworkDataContainer<float>& outputs) abstract;
	};

	class RootMeanSquareError : public AbstractErrorSystem {
	public:
		RootMeanSquareError(AbstractLayerNetwork* network) : AbstractErrorSystem(network) {}
		virtual float calculateNetworkError(const NetworkDataContainer<float>& outputs) override;
	};
}