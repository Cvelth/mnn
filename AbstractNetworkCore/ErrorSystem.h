#pragma once
#include "NetworkContainer.hpp"

namespace mnn {
	class AbstractLayerNetwork;
}

namespace mnn {
	namespace ErrorSystems {
		class AbstractErrorSystem {
		protected:
			virtual float calculate(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) abstract;
		public:
			AbstractErrorSystem() {}
			float calculateNetworkError(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs);
		};

		class MeanSquareError : public AbstractErrorSystem {
		protected:
			virtual float calculate(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) override;
		};

		class RootMeanSquareError : public AbstractErrorSystem {
		protected:
			virtual float calculate(AbstractLayerNetwork* network, const NetworkDataContainer<float>& outputs) override;
		};
	}
}