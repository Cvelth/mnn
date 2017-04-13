#pragma once
#include <functional>

namespace MNN {
	struct Link;

	class AbstractNeuron {
	private:
		float m_value;
		bool m_isValuated;
	protected:
		virtual void calculate() abstract;
		virtual float normalize(const float& value) abstract;
	public:
		AbstractNeuron(const float& value) : m_isValuated(true), m_value(value) {}
		AbstractNeuron() : m_isValuated(false) {}
		inline virtual void addInput(AbstractNeuron* i, float weight = 1.f) abstract;

		inline const float& value() {
			if (!m_isValuated)
				calculate();
			return m_value;
		}
		inline void setValueUnnormalized(const float& value) {
			m_value = value;
			m_isValuated = true;
		}
		inline void setValue(const float& value) {
			m_value = normalize(value);
			m_isValuated = true;
		}

		inline void changed() {
			m_isValuated = false;
		}
		inline bool isValuated() {
			return m_isValuated;
		}

		inline virtual void for_each(std::function<void(Link&)> lambda) abstract;
	};
}