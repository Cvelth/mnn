#pragma once
#include <functional>

namespace MNN {
	struct Link;
	class AbstractLayer;

	struct NeuronConstants {
		float eta, alpha;
		NeuronConstants(float e, float a) : eta(e), alpha(a) {}
	};
}
namespace MNN {
	class AbstractNeuron {
	private:
		float m_value;
		bool m_isValuated;
	protected:
		float m_gradient;
		NeuronConstants m_constants;
	protected:
		virtual void calculate() abstract;
		virtual float normalize(const float& value) abstract;
	public:
		AbstractNeuron(const float& value, NeuronConstants c = NeuronConstants(0.15f, 0.5f)) 
			: m_isValuated(true), m_value(value), m_constants(c) {}
		AbstractNeuron(NeuronConstants c = NeuronConstants(0.15f, 0.5f)) 
			: m_isValuated(false), m_constants(c) {}
		inline virtual void addInput(AbstractNeuron* i, float weight = 1.f) abstract;

		inline const float& value() {
			if (!m_isValuated)
				calculate();
			return m_value;
		}
		inline virtual float gradient() const {
			return m_gradient;
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

		virtual void calculateGradient(const float expectedValue) abstract;
		virtual void calculateGradient(AbstractLayer* nextLayer) abstract;
		virtual void recalculateWeights() abstract;
		virtual float getWeightTo(AbstractNeuron* neuron) abstract;

		inline virtual void for_each(std::function<void(Link&)> lambda, bool firstToLast = true) abstract;
	};
}