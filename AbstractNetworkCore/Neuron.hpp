#pragma once

#include <initializer_list>
#include <vector>
	namespace std {
		template<class _Elem> class initializer_list;
	}
#define Container std::vector

namespace MNN {
	template <typename T>
	class AbstractNeuron {
	private:
		T m_value;
		bool m_isValuated;
	protected:
		Container<T> m_inputs;
	protected:
		virtual T& calculate() abstract;
		static T& normalize(T& input) {
			return input;
		}
	public:
		AbstractNeuron(const T& value) : m_isValuated(true), m_value(value) {}
		AbstractNeuron() : m_isValuated(false) {}

		void setInputs(const std::initializer_list<T>& l) {
			m_inputs.clear();
			m_inputs.reserve(l.size());
			for (T t : l) m_inputs.push_back(t);
		}

		T value() {
			if (m_isValuated)
				return m_value;
			else {
				m_isValuated = true;
				return m_value = normalize(calculate());
			}
		}
	};

	class Neuron : public AbstractNeuron<float> {
	protected:
		virtual float& calculate() override {
			float result = 0;
			for (float t : m_inputs)
				result += t;
			return result;
		}
	};
}