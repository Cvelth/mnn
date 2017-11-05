#pragma once
#include "Shared.hpp"
#include <functional>
namespace mnn {
	class AbstractNeuron;
	class AbstractLayer {
	public:
		AbstractLayer() {}
		~AbstractLayer() {}
		inline virtual void add(AbstractNeuron* i) =0;
		inline virtual size_t size() const =0;
		inline virtual Type at(size_t index) const =0;
		inline Type operator[](size_t index) const { return at(index); }
		virtual std::string print() const =0;

		inline virtual void calculate() =0;

		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const =0;

		friend std::istream& operator>>(std::istream &s, AbstractLayer *&n);
	};
}