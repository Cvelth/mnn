#pragma once
#include <functional>
#include "Shared.hpp"
namespace mnn {
	struct Link;
	class AbstractLayer;
	class AbstractNeuron {
	private:
		Type m_value;
		bool m_isValuated;

		size_t m_id;
		static size_t NUMBER_OF_NEURONS_CREATED;
	protected:
		Type m_gradient;
		Type m_eta;
		Type m_alpha;
	protected:
		virtual void calculate() =0;
		virtual Type normalize(Type const& value) =0;
	public:
		AbstractNeuron(Type const& value, Type const& eta, Type const& alpha) : m_isValuated(true),
			m_value(value), m_eta(eta), m_alpha(alpha), m_id(NUMBER_OF_NEURONS_CREATED++) {}
		AbstractNeuron(Type const& eta, Type const& alpha) : m_isValuated(false),
			m_eta(eta), m_alpha(alpha), m_id(NUMBER_OF_NEURONS_CREATED++) {}
		virtual ~AbstractNeuron() {};
		virtual void link(AbstractNeuron *i, Type const& weight = 1.f) =0;
		virtual void link(Link const& l) =0;
		virtual void link(LinkContainer<Link> const& l) =0;
		inline const Type& value() {
			if (!m_isValuated)
				calculate();
			return m_value;
		}
		inline virtual Type const& gradient() const { return m_gradient; }
		inline void setValueUnnormalized(Type const& value) {
			m_value = value;
			m_isValuated = true;
		}
		inline void setValue(Type const& value) {
			m_value = normalize(value);
			m_isValuated = true;
		}
		inline void changed() { m_isValuated = false; }
		inline bool isValuated() { return m_isValuated; }

		virtual void calculateGradient(Type const& expectedValue) =0;
		[[deprecated]] virtual void calculateGradient(AbstractLayer* nextLayer) =0;
		virtual void recalculateWeights() =0;
		virtual Type getWeightTo(AbstractNeuron* neuron) =0;

		inline virtual void for_each_link(std::function<void(Link&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each_link(std::function<void(Link const&)> lambda, bool firstToLast = true) const =0;
	};
}