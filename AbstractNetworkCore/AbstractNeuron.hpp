#pragma once

namespace MNN {
	template <typename T>
	class AbstractNeuron {
	private:
		T m_value;
		bool m_isValuated;
	protected:
		virtual void calculate() abstract;
		virtual T normalize(const T& value) abstract;
	public:
		AbstractNeuron(const T& value) : m_isValuated(true), m_value(value) {}
		AbstractNeuron() : m_isValuated(false) {}
		inline virtual void addInput(AbstractNeuron<T>* i) abstract;

		inline T value() {
			if (!m_isValuated)
				calculate();
			return m_value;
		}
		inline void setValue(const T& value) {
			m_value = normalize(value);
			m_isValuated = true;
		}

		inline void changed() {
			m_isValuated = false;
		}
		inline bool isValuated() {
			return m_isValuated;
		}

	};
}