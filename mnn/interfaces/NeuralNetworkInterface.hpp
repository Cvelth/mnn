#pragma once
#include "mnn/interfaces/Types.hpp"
namespace mnn {
	class NeuralNetworkInterface {
	public:
		virtual ~NeuralNetworkInterface() {}

		virtual NeuronContainer<Value>& inputs() = 0;
		virtual NeuronContainer<Value> const& inputs() const = 0;
		virtual NeuronContainer<Value>& outputs() = 0;
		virtual NeuronContainer<Value> const& outputs() const = 0;

		virtual void inputs(NeuronContainer<Value> const& _inputs, bool normalize = true) = 0;

		virtual void process() = 0;
		inline void process(NeuronContainer<Value> const& _inputs, bool normalize = true) {
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
		virtual void updateWeights() = 0;
		inline void backpropagate(NeuronContainer<Value> const& _outputs) {
			calculateGradients(_outputs);
			updateWeights();
		}
	};
}