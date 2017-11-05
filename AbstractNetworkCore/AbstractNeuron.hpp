#pragma once
#include <functional>
#include "Shared.hpp"
namespace mnn {
	struct Link;
	class AbstractNeuron {
	private:
		Type m_value;
		bool m_isValuated;

		size_t m_id;
		static size_t NUMBER_OF_NEURONS_CREATED;
	protected:
		virtual void calculate() =0;
		virtual Type normalize(Type const& value) =0;
	public:
		AbstractNeuron() : m_isValuated(false), m_id(NUMBER_OF_NEURONS_CREATED++) {}
		AbstractNeuron(Type const& value) : m_isValuated(true),	m_value(value), 
			m_id(NUMBER_OF_NEURONS_CREATED++) {}
		virtual ~AbstractNeuron() {};

		inline size_t id() const { return m_id; }
		static size_t next_id() { return NUMBER_OF_NEURONS_CREATED; }
		virtual void link(AbstractNeuron *i, Type const& weight = 1.f) =0;
		virtual void link(Link const& l) =0;
		virtual void link(LinkContainer<Link> const& l) =0;
		virtual LinkContainer<Link> const& links() const = 0;
		inline virtual void clear_links() =0;
		inline virtual void update_links(LinkContainer<Link> const& c) =0;
		inline const Type& value() {
			if (!m_isValuated)
				calculate();
			return m_value;
		}
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
		virtual std::string print() const =0;

		inline virtual void for_each_link(std::function<void(Link&)> lambda, bool firstToLast = true) =0;
		inline virtual void for_each_link(std::function<void(Link const&)> lambda, bool firstToLast = true) const =0;

		friend std::istream& operator>>(std::istream &s, AbstractNeuron *&n);
	};
	struct BackpropagationLink;
	class AbstractBackpropagationNeuron : public AbstractNeuron {
	protected:
		Type m_gradient;
		Type m_eta;
		Type m_alpha;
	public:
		AbstractBackpropagationNeuron(Type const& value, Type const& eta, Type const& alpha) 
			: AbstractNeuron(value), m_eta(eta), m_alpha(alpha) {}
		AbstractBackpropagationNeuron(Type const& eta, Type const& alpha)
			: AbstractNeuron(), m_eta(eta), m_alpha(alpha) {}

		inline virtual Type const& gradient() const { return m_gradient; }
		virtual void calculateGradient(Type const& expectedValue) = 0;
		virtual void calculateGradient(std::function<Type(std::function<Type(AbstractBackpropagationNeuron&)>)> gradient_sum) = 0;
		virtual void recalculateWeights() = 0;
		virtual Type getWeightTo(AbstractBackpropagationNeuron* neuron) = 0;

		inline virtual void for_each_link(std::function<void(BackpropagationLink&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_link(std::function<void(BackpropagationLink const&)> lambda, bool firstToLast = true) const = 0;

		friend std::istream& operator>>(std::istream &s, AbstractNeuron *&n);
	};
}