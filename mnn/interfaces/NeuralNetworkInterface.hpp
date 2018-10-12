#pragma once
#include "mnn/interfaces/Types.hpp"
#include "mnn/interfaces/NeuronInterface.hpp"
namespace mnn {
	class NeuralNetworkInterface {
	protected:
		NeuronContainer<std::shared_ptr<NeuronInterface>> m_inputs;
		NeuronContainer<std::shared_ptr<NeuronInterface>> m_outputs;
	public:
		virtual ~NeuralNetworkInterface() {}

		inline NeuronContainer<std::shared_ptr<NeuronInterface>> inputs() { return m_inputs; }
		inline NeuronContainer<std::shared_ptr<NeuronInterface>> const& inputs() const { return m_inputs; }
		inline NeuronContainer<std::shared_ptr<NeuronInterface>> outputs() { return m_outputs; }
		inline NeuronContainer<std::shared_ptr<NeuronInterface>> const& outputs() const { return m_outputs; }

		inline void inputs(NeuronContainer<Value> const& _inputs, bool normalize = true) {
			auto it1 = m_inputs.begin();
			auto it2 = _inputs.cbegin();
			while (it1 != m_inputs.end() || it2 != _inputs.cend()) {
				(*it1)->value(*it2, normalize);
				it1++; it2++;
			}
		}
		inline void inputs(NeuronContainer<std::shared_ptr<NeuronInterface>> const& _inputs, bool normalize = true) {
			auto it1 = m_inputs.begin();
			auto it2 = _inputs.cbegin();
			while (it1 != m_inputs.end() || it2 != _inputs.cend()) {
				(*it1)->value(**it2, normalize);
				it1++; it2++;
			}
		}

		virtual void process() = 0;
		inline void process(NeuronContainer<Value> const& _inputs, bool normalize = true) {
			inputs(_inputs, normalize);
			process();
		}
		inline void process(NeuronContainer<std::shared_ptr<NeuronInterface>> const& _inputs, bool normalize = true) {
			inputs(_inputs, normalize);
			process();
		}

		/* TO DO
		friend std::ostream& operator<<(std::ostream &s, NetworkInterface const* n);
		friend std::istream& operator>>(std::istream &s, NetworkInterface *&n);
		*/
	};

	class BackpropagationNeuralNetworkInterface : public NeuralNetworkInterface {
	public:
		virtual void calculateGradients(NeuronContainer<Value> const& _outputs) = 0;
		virtual void calculateGradients(NeuronContainer<std::shared_ptr<NeuronInterface>> const& _inputs) = 0;
		virtual void updateWeights() = 0;
		inline void backpropagate(NeuronContainer<Value> const& _outputs) {
			calculateGradients(_outputs);
			updateWeights();
		}
		inline void backpropagate(NeuronContainer<std::shared_ptr<NeuronInterface>> const& _outputs) {
			calculateGradients(_outputs);
			updateWeights();
		}
	};
}