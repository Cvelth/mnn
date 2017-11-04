#pragma once
#include <functional>
namespace mnn {
	class AbstractNeuron;
	class AbstractLayer {
	public:
		AbstractLayer() {}
		~AbstractLayer() {}
		inline virtual void add(AbstractNeuron* i) =0;
		inline virtual size_t size() const =0;

		inline virtual void calculate() =0;

		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const =0;
	};
}