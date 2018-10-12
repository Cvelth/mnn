#pragma once
#include <functional>
#include <memory>
#include "mnn/interfaces/Types.hpp"
namespace mnn {
	struct ExplicitLink;
	class NeuronInterface {
	private:
		Value m_value;
		bool m_isEvaluated;

		size_t m_id;
		static size_t NUMBER_OF_NEURONS_CREATED;
	protected:
		virtual void calculate(bool full = false) = 0;
		virtual bool is_dependent() const = 0;
		static Value normalize(Value const& value);
		static Value normalization_derivative(Value const& value);
	public:
		NeuronInterface() : m_isEvaluated(false), m_id(NUMBER_OF_NEURONS_CREATED++) {}
		NeuronInterface(Value const& value) : m_isEvaluated(true), m_value(value),
			m_id(NUMBER_OF_NEURONS_CREATED++) {}
		virtual ~NeuronInterface() {};

		NeuronInterface& operator=(Value const& value) {
			m_isEvaluated = true; m_value = value;
		}
		bool operator==(NeuronInterface const& other) const { return m_id == other.m_id; }

		inline size_t const& id() const { return m_id; }
		static size_t const& next_id() { return NUMBER_OF_NEURONS_CREATED; }

		virtual void link(std::shared_ptr<NeuronInterface> i, Value const& weight = 1.f) = 0;
		inline virtual void clear_links() = 0;

		inline Value const& value(bool full_recalculation = false) {
			if ((full_recalculation && is_dependent()) || !m_isEvaluated)
				calculate(full_recalculation);
			return m_value;
		}
		inline Value const& operator*() { return value(); }
		inline Value const& operator()() { return value(); }
		inline operator Value() { return value(); }
		inline void value(Value const& value, bool to_normalize = true) {
			m_value = to_normalize ? normalize(value) : value;
			m_isEvaluated = true;
		}

		inline void changed() { m_isEvaluated = false; }
		inline bool isValuated() { return m_isEvaluated; }

		/* Unimplemented from v1.0
		virtual LinkContainer<ExplicitLink> links() const = 0;
		inline virtual void update_links(LinkContainer<Link> const& c) = 0;
		
		virtual std::string print() const = 0;

		inline virtual void for_each_link(std::function<void(Link&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_link(std::function<void(Link const&)> lambda, bool firstToLast = true) const = 0;

		friend std::istream& operator>>(std::istream &s, AbstractNeuron *&n);
		*/
	};

	struct ExplicitBackpropagationLink;
	class BackpropagationNeuronInterface : public NeuronInterface {
	protected:
		Value m_gradient;
		Value m_eta, m_alpha;
	public:
		BackpropagationNeuronInterface(Value const& value, Value const& eta, Value const& alpha)
			: NeuronInterface(value), m_eta(eta), m_alpha(alpha) {}
		BackpropagationNeuronInterface(Value const& eta, Value const& alpha)
			: NeuronInterface(), m_eta(eta), m_alpha(alpha) {}

		Value const& eta() const { return m_eta; }
		Value const& alpha() const { return m_alpha; }
		void eta(Value const& value) { m_eta = value; }
		void alpha(Value const& value) { m_alpha = value; }

		inline virtual Value const& gradient() const { return m_gradient; }

		virtual void calculateGradient(Value const& expectedValue) = 0;
		virtual void calculateGradient(std::function<Value(std::function<Value(BackpropagationNeuronInterface&)>)> gradient_sum) = 0;
		virtual void recalculateWeights() = 0;
		virtual Value getWeightTo(BackpropagationNeuronInterface *neuron) = 0;

		/* Unimplemented from v1.0
		inline virtual void for_each_link(std::function<void(BackpropagationLink&)> lambda, bool firstToLast = true) = 0;
		inline virtual void for_each_link(std::function<void(BackpropagationLink const&)> lambda, bool firstToLast = true) const = 0;

		friend std::istream& operator>>(std::istream &s, AbstractNeuron *&n);
		*/
	};
}