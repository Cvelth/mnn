#pragma once
#include "AbstractNeuron.hpp"

namespace MNN {
	template <typename T>
	class AbstractLayer {
	private:

	protected:

	public:
		AbstractLayer() {}
		inline virtual void add(AbstractNeuron<T>* i) abstract;
		inline virtual void remove(AbstractNeuron<T>* i) abstract;

		inline virtual void calculate() abstract;
	};
}