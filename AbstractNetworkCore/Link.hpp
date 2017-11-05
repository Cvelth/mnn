#pragma once
#include "Shared.hpp"
#include <functional>
#include <algorithm>
namespace mnn {
	class AbstractNeuron;
	struct Link {
		AbstractNeuron* unit;
		Type weight;
		Link(AbstractNeuron* unit, Type const& weight) : unit(unit), weight(weight) {}
	};
	struct BackpropagationLink : public Link {
		Type delta;
		BackpropagationLink(AbstractNeuron* unit, Type const& weight, Type const& delta = 0.f) 
			: Link(unit, weight), delta(delta) {}
		BackpropagationLink(Link const& other, Type const& delta = 0.f) 
			: Link(other.unit, other.weight), delta(delta) {}
		inline void step() { weight += delta; }
		operator Link() const { return Link(unit, weight); }
	};
	template <typename LinkType>
	class LinkStorage {
	protected:
		LinkContainer<LinkType> m_links;
	public:
		LinkContainer<LinkType>& operator*() { return m_links; }
		LinkContainer<LinkType> const& operator*() const { return m_links; }
		LinkContainer<LinkType>* operator->() { return &m_links; }
		LinkContainer<LinkType> const* operator->() const { return &m_links; }
		inline void for_each(std::function<void(LinkType&)> lambda, bool firstToLast = true) {
			if (firstToLast)
				for (auto it = m_links.begin(); it != m_links.end(); it++)
					lambda(*it);
			else
				for (auto it = m_links.rbegin(); it != m_links.rend(); it++)
					lambda(*it);
		}
		inline void for_each(std::function<void(LinkType const&)> lambda, bool firstToLast = true) const {
			if (firstToLast)
				for (auto it = m_links.begin(); it != m_links.end(); it++)
					lambda(*it);
			else
				for (auto it = m_links.rbegin(); it != m_links.rend(); it++)
					lambda(*it);
		}
		template <typename ConvertableLinkType>
		operator LinkContainer<ConvertableLinkType>() const {
			LinkContainer<ConvertableLinkType> res;
			std::copy(m_links.begin(), m_links.end(), res.begin());
			return res;
		}
	};
}