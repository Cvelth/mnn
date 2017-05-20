#pragma once
#include "NetworkContainer.hpp"

namespace MNN {
	class AbstractLayerNetwork;
}

namespace MNN {
	class AbstractErrorSystem {
	public:
		AbstractErrorSystem() {}
		virtual float calculateNetworkError(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) abstract;
	};

	class RootMeanSquareError : public AbstractErrorSystem {
	public:
		RootMeanSquareError() : AbstractErrorSystem() {}
		virtual float calculateNetworkError(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) override;
	};
}