#pragma once
#include "mnn/interfaces/NeuronInterface.hpp"
#include "mnn/interfaces/ExplicitLink.hpp"
namespace mnn {
	class Neuron : public NeuronInterface {
	protected:
		LinkContainer<ExplicitLink> m_links;
		virtual void calculate(bool full = false) override;
		virtual bool is_dependent() const override { return !m_links.empty(); }
	public:
		Neuron() : NeuronInterface() {}
		Neuron(Value const& value) : NeuronInterface(value) {}
		virtual ~Neuron() {}

		virtual void link(std::shared_ptr<NeuronInterface> i, Value const& weight = 1.f) override {
			m_links.push_back(ExplicitLink(i, weight));
			changed();
		}
		inline virtual void clear_links() override { m_links.clear(); changed(); }
	};
	class BackpropagationNeuron : public BackpropagationNeuronInterface {
	protected:
		LinkContainer<ExplicitBackpropagationLink> m_links;
		virtual void calculate(bool full = false) override;
		virtual bool is_dependent() const override { return !m_links.empty(); }
	public:
		BackpropagationNeuron() : BackpropagationNeuronInterface(0.15, 0.5) {}
		BackpropagationNeuron(Value const& value)
			: BackpropagationNeuronInterface(value, 0.15, 0.5) {}
		BackpropagationNeuron(Value const& eta, Value const& alpha)
			: BackpropagationNeuronInterface(eta, alpha) {}
		BackpropagationNeuron(Value const& value, Value const& eta, Value const& alpha)
			: BackpropagationNeuronInterface(value, eta, alpha) {}

		virtual void calculateGradient(Value const& expectedValue) override;
		virtual void calculateGradient(std::function<Value(std::function<Value(BackpropagationNeuronInterface&)>)> gradient_sum) override;
		virtual Value getWeightTo(BackpropagationNeuronInterface *neuron) override;
		virtual void recalculateWeights() override;

		virtual void link(std::shared_ptr<NeuronInterface> i, Value const& weight = 1.f) override {
			m_links.push_back(ExplicitBackpropagationLink(i, weight));
			changed();
		}
		inline virtual void clear_links() override { m_links.clear(); changed(); }
	};
}