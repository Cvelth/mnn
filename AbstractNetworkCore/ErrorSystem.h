#pragma once
#include "NetworkContainer.hpp"

namespace MNN {
	class AbstractLayerNetwork;
}

namespace MNN {
	class AbstractErrorSystem {
	protected:
		virtual float calculate(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) abstract;
	public:
		AbstractErrorSystem() {}
		virtual float calculateNetworkError(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) {
			if (outputs.size() != network->getOutputsNumber())
				throw Exceptions::WrongOutputNumberException();
			return calculate(network, outputs);
		}
	};

	class MeanSquareError : public AbstractErrorSystem {
	protected:
		virtual float calculate(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) abstract;
	};

	class RootMeanSquareError : public AbstractErrorSystem {
	protected:
		virtual float calculate(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) abstract;
	};
}