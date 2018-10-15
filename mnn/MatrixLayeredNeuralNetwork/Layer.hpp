#pragma once
#include <memory>
#include "mnn/interfaces/Types.hpp"
#include "mnn/exceptions.hpp"
DefineNewMNNException(UnsupportedInputError);
namespace mnn {
	class Layer {
	protected:
		bool m_bias;
		NeuronContainer<Value> m_value;
		NeuronContainer<NeuronContainer<Value>> m_weights;
	protected:
		virtual std::ostream& to_stream(std::ostream &output) const;
		virtual std::istream& from_stream(std::istream &input);

		Layer() {}
	public:
		Layer(size_t const& size, size_t const& input_number, bool bias = true, Value const& minimum_weight_value = 0.0, Value const& maximum_weight_value = 1.0);

		inline size_t size() const { return m_weights.at(0).size(); }
		inline size_t input_number() const { return m_weights.size(); }

		NeuronContainer<Value> process(NeuronContainer<Value> const& inputs);
		NeuronContainer<Value> const& value() const { return m_value; }

		friend std::ostream& operator<<(std::ostream &s, Layer const& l) {
			return l.to_stream(s);
		}
		friend std::istream& operator>>(std::istream &s, Layer &l) {
			return l.from_stream(s);
		}
		static std::shared_ptr<Layer> read(std::istream &s) {
			auto ret = new Layer();
			ret->from_stream(s);
			return std::shared_ptr<Layer>(ret);
		}
	};

	class MatrixLayeredBackpropagationNeuralNetwork;
	class BackpropagationLayer : public Layer {
		friend MatrixLayeredBackpropagationNeuralNetwork;
	protected:
		NeuronContainer<NeuronContainer<Value>> m_deltas;
	protected:
		virtual std::ostream& to_stream(std::ostream &output) const override;
		virtual std::istream& from_stream(std::istream &input) override;

		BackpropagationLayer() {}
	public:
		BackpropagationLayer(size_t const& size, size_t const& input_number, bool bias = true, Value const& minimum_weight_value = 0.0, Value const& maximum_weight_value = 1.0);
		static std::shared_ptr<BackpropagationLayer> read(std::istream &s) {
			auto ret = new BackpropagationLayer();
			ret->from_stream(s);
			return std::shared_ptr<BackpropagationLayer>(ret);
		}
	};
}