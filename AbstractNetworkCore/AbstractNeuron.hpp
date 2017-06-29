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
	//Abstract class for storing typical neuron data to be inherited by other classes. 
	//Handles inner calculation of the network parts for value generation and learning.
	class AbstractNeuron {
	private:
		float m_value;
		bool m_isValuated;
	protected:
		float m_gradient;
		NeuronConstants m_constants;
	protected:
		//Run the calculation process with inserted inputs.
		virtual void calculate() abstract;
		//Returns normalized value of parameter *value*.
		virtual float normalize(const float& value) abstract;
	public:
		//Constructs static neuron(with inserted constant value).
		AbstractNeuron(const float& value, NeuronConstants c = NeuronConstants(0.15f, 0.5f)) 
			: m_isValuated(true), m_value(value), m_constants(c) {}
		//Constructs active neuron to be connected to others.
		AbstractNeuron(NeuronConstants c = NeuronConstants(0.15f, 0.5f)) 
			: m_isValuated(false), m_constants(c) {}
		//Adds one more input neuron reference.
		inline virtual void addInput(AbstractNeuron* i, float weight = 1.f) abstract;

		//Returns value of the neuron. 
		//If it isn't calculated, runs calculation process.
		inline const float& value() {
			if (!m_isValuated)
				calculate();
			return m_value;
		}
		//Returns the gradiens.
		inline virtual float gradient() const {
			return m_gradient;
		}
		//Set neuron's value to a constant without normalization.
		inline void setValueUnnormalized(const float& value) {
			m_value = value;
			m_isValuated = true;
		}
		//Set neuron's value to a constant with normalization of the value.
		inline void setValue(const float& value) {
			m_value = normalize(value);
			m_isValuated = true;
		}
		//Tells neuron to recalculate value on the next *value* function call.
		inline void changed() {
			m_isValuated = false;
		}
		//Checks whether Neuron will recalculate value on the next *value* function call.
		inline bool isValuated() {
			return m_isValuated;
		}

		//Calculates gradient accordingly to the expected value.
		virtual void calculateGradient(const float expectedValue) abstract;
		//Calculates gradient accordingly to the next layer.
		virtual void calculateGradient(AbstractLayer* nextLayer) abstract;
		//Recalculates all the weights of links.
		virtual void recalculateWeights() abstract;
		//Returns value of the weight between this neuron and the one passed by *neuron*.
		virtual float getWeightTo(AbstractNeuron* neuron) abstract;

		//Executes *lambda* for every link.
		inline virtual void for_each(std::function<void(Link&)> lambda, bool firstToLast = true) abstract;
	};
}