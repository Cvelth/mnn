#pragma once
#include "Shared.hpp"
#include "AbstractNeuron.hpp"
#include "Link.hpp"
namespace mnn {
	class AbstractLayer;
	class Neuron : public AbstractNeuron {
	protected:
		LinkContainer<Link> m_links;
		virtual void calculate() override;
		virtual Type normalize(const Type& value) override;
	public:
		Neuron(Type const& value, Type const& eta, Type const& alpha)
			: AbstractNeuron(value, eta, alpha) {}
		Neuron(Type const& eta, Type const& alpha)
			: AbstractNeuron(eta, alpha) {}
		Neuron(const LinkContainer<Link>& c, Type const& eta, Type const& alpha)
			: AbstractNeuron(eta, alpha) { link(c); }
		virtual ~Neuron() {}
		virtual void link(AbstractNeuron *i, Type const& weight = 1.f) override {
			link(Link(i, weight));
		}
		virtual void link(Link const& l) override {
			m_links.push_back(l);
			changed();
		}
		virtual void link(LinkContainer<Link> const& l) override {
			for (auto it : l) link(it);
		}
		inline virtual LinkContainer<Link> const& links() const override { return m_links; }
		inline virtual void clear_links() override { m_links.clear(); }
		inline virtual void update_links(LinkContainer<Link> const& c) override {
			clear_links();
			m_links.reserve(c.size());
			link(c);
		}
		virtual void calculateGradient(Type const& expectedValue) override;
		virtual void calculateGradient(std::function<Type(std::function<Type(AbstractNeuron&)>)> gradient_sum) override;
		virtual Type getWeightTo(AbstractNeuron* neuron) override;
		virtual void recalculateWeights() override;

		virtual std::string print() const override;

		inline virtual void for_each_link(std::function<void(Link&)> lambda, bool firstToLast = true) override {
			if (firstToLast)
				for (auto it = m_links.begin(); it != m_links.end(); it++)
					lambda(*it);
			else
				for (auto it = m_links.rbegin(); it != m_links.rend(); it++)
					lambda(*it);
		}
		inline virtual void for_each_link(std::function<void(Link const&)> lambda, bool firstToLast = true) const override{
			if (firstToLast)
				for (auto it = m_links.begin(); it != m_links.end(); it++)
					lambda(*it);
			else
				for (auto it = m_links.rbegin(); it != m_links.rend(); it++)
					lambda(*it);
		}
	};
}