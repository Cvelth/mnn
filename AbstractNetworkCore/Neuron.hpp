#pragma once
#include "Shared.hpp"
#include "AbstractNeuron.hpp"
#include "Link.hpp"
namespace mnn {
	class Neuron : public AbstractNeuron {
	protected:
		LinkStorage<Link> m_links;
		virtual void calculate() override;
		virtual Type normalize(const Type& value) override;
	public:
		Neuron() : AbstractNeuron() {}
		Neuron(Type const& value) : AbstractNeuron(value) {}
		Neuron(const LinkContainer<Link>& c) : AbstractNeuron() { link(c); }
		virtual ~Neuron() {}
		virtual void link(AbstractNeuron *i, Type const& weight = 1.f) override {
			link(Link(i, weight));
		}
		virtual void link(Link const& l) override {
			m_links->push_back(l);
			changed();
		}
		virtual void link(LinkContainer<Link> const& l) override {
			for (auto it : l) link(it);
		}
		inline virtual LinkContainer<Link> const& links() const override { return *m_links; }
		inline virtual void clear_links() override { m_links->clear(); }
		inline virtual void update_links(LinkContainer<Link> const& c) override {
			clear_links();
			m_links->reserve(c.size());
			link(c);
		}
		inline virtual void for_each_link(std::function<void(Link&)> lambda, bool firstToLast = true) override {
			m_links.for_each(lambda, true);
		}
		inline virtual void for_each_link(std::function<void(Link const&)> lambda, bool firstToLast = true) const override {
			m_links.for_each(lambda, true);
		}

		virtual std::string print() const override;
	};
	class BackpropagationNeuron : public AbstractBackpropagationNeuron {
	protected:
		LinkStorage<BackpropagationLink> m_links;
		virtual void calculate() override;
		virtual Type normalize(const Type& value) override;
	public:
		BackpropagationNeuron(Type const& value, Type const& eta, Type const& alpha)
			: AbstractBackpropagationNeuron(value, eta, alpha) {}
		BackpropagationNeuron(Type const& eta, Type const& alpha)
			: AbstractBackpropagationNeuron(eta, alpha) {}
		BackpropagationNeuron(const LinkContainer<Link>& c, Type const& eta, Type const& alpha)
			: AbstractBackpropagationNeuron(eta, alpha) { link(c); }

		virtual void calculateGradient(Type const& expectedValue) override;
		virtual void calculateGradient(std::function<Type(std::function<Type(AbstractBackpropagationNeuron&)>)> gradient_sum) override;
		virtual Type getWeightTo(AbstractBackpropagationNeuron* neuron) override;
		virtual void recalculateWeights() override;

		virtual void link(AbstractNeuron *i, Type const& weight = 1.f) override {
			link(Link(i, weight));
		}
		virtual void link(Link const& l) override {
			m_links->push_back(l);
			changed();
		}
		virtual void link(LinkContainer<Link> const& l) override {
			for (auto it : l) link(it);
		}
		inline virtual LinkContainer<Link> const& links() const override { return m_links; }
		inline virtual void clear_links() override { m_links->clear(); }
		inline virtual void update_links(LinkContainer<Link> const& c) override {
			clear_links();
			m_links->reserve(c.size());
			link(c);
		}
		inline virtual void for_each_link(std::function<void(Link&)> lambda, bool firstToLast = true) override {
			m_links.for_each(lambda, true);
		}
		inline virtual void for_each_link(std::function<void(Link const&)> lambda, bool firstToLast = true) const override {
			m_links.for_each(lambda, true);
		}
		inline virtual void for_each_link(std::function<void(BackpropagationLink&)> lambda, bool firstToLast = true) override {
			m_links.for_each(lambda, true);
		}
		inline virtual void for_each_link(std::function<void(BackpropagationLink const&)> lambda, bool firstToLast = true) const override {
			m_links.for_each(lambda, true);
		}

		virtual std::string print() const override;
	};
}