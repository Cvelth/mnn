#pragma once
#include <functional>

namespace MNN {
	class AbstractNeuron;

	class AbstractLayer {
	private:

	protected:

	public:
		AbstractLayer() {}
		inline virtual void add(AbstractNeuron* i) abstract;
		inline virtual void remove(AbstractNeuron* i) abstract;
		inline virtual size_t size() const abstract;

		inline virtual void calculate() abstract;

		inline virtual void for_each(std::function<void(AbstractNeuron*)> lambda) abstract;
	};
}