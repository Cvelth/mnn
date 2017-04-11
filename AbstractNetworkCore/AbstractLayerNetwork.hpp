#pragma once
#include "AbstractNetwork.hpp"
#include "AbstractLayer.hpp"

namespace MNN {
	template <typename T>
	class AbstractLayerNetwork : public AbstractNetwork<T> {
	private:

	protected:
		inline virtual void addLayer(AbstractLayer<T>* l) abstract;
	public:
		virtual void calculate() abstract;
	};
}