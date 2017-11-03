#pragma once
#include <functional>
namespace mnn {
	class AbstractNeuron;
	class AbstractLayer {
	public:
		AbstractLayer() {}
		~AbstractLayer() {}
		inline virtual void add(AbstractNeuron* i) abstract;
		inline virtual void remove(AbstractNeuron* i) abstract;
		inline virtual size_t size() const abstract;

		inline virtual void calculate() abstract;

		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) abstract;
		inline virtual void for_each(std::function<void(AbstractNeuron const&)> lambda, bool firstToLast = true) const abstract;
	};
}