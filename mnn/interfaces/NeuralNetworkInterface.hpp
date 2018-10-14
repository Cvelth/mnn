#pragma once
#include "mnn/interfaces/Types.hpp"
#include <string>
namespace mnn {
	class NeuralNetworkInterface {
	protected:
		NeuronContainer<Value> m_inputs;
		NeuronContainer<Value> m_outputs;
	protected:
		virtual std::ostream& to_stream(std::ostream &output) const abstract;
		virtual std::istream& from_stream(std::istream &input) abstract;
	public:
		NeuralNetworkInterface(size_t input_number, size_t output_number) {
			m_inputs.resize(input_number);
			m_outputs.resize(output_number);
		}
		virtual ~NeuralNetworkInterface() {}

		inline NeuronContainer<Value> inputs() { return m_inputs; }
		inline NeuronContainer<Value> const& inputs() const { return m_inputs; }
		inline NeuronContainer<Value> outputs() { return m_outputs; }
		inline NeuronContainer<Value> const& outputs() const { return m_outputs; }

		inline void inputs(NeuronContainer<Value> const& _inputs, bool _normalize = true) {
			auto it1 = m_inputs.begin();
			auto it2 = _inputs.cbegin();
			while (it1 != m_inputs.end() || it2 != _inputs.cend()) {
				*it1 = _normalize ? normalize(*it2) : *it2;
				it1++; it2++;
			}
		}

		virtual void process() = 0;
		inline void process(NeuronContainer<Value> const& _inputs, bool normalize = true) {
			inputs(_inputs, normalize);
			process();
		}

		friend std::ostream& operator<<(std::ostream &s, NeuralNetworkInterface const& n) {
			return n.to_stream(s);
		}
		friend std::istream& operator>>(std::istream &s, NeuralNetworkInterface &n) {
			return n.from_stream(s);
		}
	};

	class BackpropagationNeuralNetworkInterface : public NeuralNetworkInterface {
	public:
		using NeuralNetworkInterface::NeuralNetworkInterface;
		virtual void backpropagate(NeuronContainer<Value> const& _outputs) = 0;
	};
}