#pragma once
#include "Shared.hpp"
#include "AbstractLayerNetwork.hpp"
#include "LayerNetworkStorage.hpp"
namespace mnn {
	class LayerNetwork : public AbstractLayerNetwork {
	protected:
		LayerNetworkStorage<AbstractNeuron> m_layers;
	public:
		explicit LayerNetwork(AbstractLayer<AbstractNeuron>* inputs, AbstractLayer<AbstractNeuron>* outputs) : m_layers(inputs, outputs) {}
		explicit LayerNetwork(AbstractLayer<AbstractNeuron>* inputs, AbstractLayer<AbstractNeuron>* outputs, LayerContainer<AbstractLayer<AbstractNeuron>*> const& c)
			: LayerNetwork(inputs, outputs) { addHiddenLayers(c); }
		inline virtual void addHiddenLayer(AbstractLayer<AbstractNeuron>* l) override { m_layers->push_back(l); }
		inline virtual void addHiddenLayers(LayerContainer<AbstractLayer<AbstractNeuron>*> const& c) override { for (auto t : c) addHiddenLayer(t); }

		virtual void setInputs(NeuronContainer<Type> const& inputs, bool normalize = true) override;
		virtual size_t getInputsNumber() const override;
		virtual size_t getOutputsNumber() const override;
		virtual NeuronContainer<Type> getInputs() const override;
		virtual NeuronContainer<Type> getOutputs() const override;
		virtual const float getInput(size_t index) const override;
		virtual const float getOutput(size_t index) const override;
		inline virtual AbstractLayer<AbstractNeuron> const* getInputLayer() const override { return m_layers.inputs(); }
		inline virtual AbstractLayer<AbstractNeuron> const* getOutputLayer() const override { return m_layers.outputs(); }
		virtual bool check_compatibility(AbstractNetwork const* other) const override;

		inline virtual void calculate() override { m_layers.outputs()->calculate(); }
		
		inline virtual void for_each_hidden(std::function<void(AbstractLayer<AbstractNeuron>&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_hidden(lambda, firstToLast);
		}
		inline virtual void for_each_hidden(std::function<void(AbstractLayer<AbstractNeuron>&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_hidden(lambda, firstToLast);
		}
		inline virtual void for_each_layer(std::function<void(AbstractLayer<AbstractNeuron>&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_layer(lambda, firstToLast);
		}
		inline virtual void for_each_layer(std::function<void(AbstractLayer<AbstractNeuron>&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_layer(lambda, firstToLast);
		}
		inline virtual void for_each_input(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_input(lambda, firstToLast);
		}
		inline virtual void for_each_input(std::function<void(AbstractNeuron&)> lambda, bool const firstToLast = true) const override {
			m_layers.for_each_input(lambda, firstToLast);
		}
		inline virtual void for_each_output(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_output(lambda, firstToLast);
		}
		inline virtual void for_each_output(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_output(lambda, firstToLast);
		}
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_hidden_neuron(lambda, firstToLast);
		}
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_hidden_neuron(lambda, firstToLast);
		}
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_neuron(lambda, firstToLast);
		}
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_neuron(lambda, firstToLast);
		}

		virtual std::string print() const override;
	};

	class BackpropagationLayerNetwork : public AbstractBackpropagationLayerNetwork {
	protected:
		LayerNetworkStorage<AbstractBackpropagationNeuron> m_layers;
	public:
		explicit BackpropagationLayerNetwork(AbstractLayer<AbstractBackpropagationNeuron>* inputs, AbstractLayer<AbstractBackpropagationNeuron>* outputs) : m_layers(inputs, outputs) {}
		explicit BackpropagationLayerNetwork(AbstractLayer<AbstractBackpropagationNeuron>* inputs, AbstractLayer<AbstractBackpropagationNeuron>* outputs, LayerContainer<AbstractLayer<AbstractBackpropagationNeuron>*> const& c)
			: BackpropagationLayerNetwork(inputs, outputs) { addHiddenLayers(c); }
		inline virtual void addHiddenLayer(AbstractLayer<AbstractBackpropagationNeuron>* l) override { m_layers->push_back(l); }
		inline virtual void addHiddenLayers(LayerContainer<AbstractLayer<AbstractBackpropagationNeuron>*> const& c) override { for (auto t : c) addHiddenLayer(t); }

		virtual void setInputs(NeuronContainer<Type> const& inputs, bool normalize = true) override;
		void calculateGradients(const NeuronContainer<Type>& outputs);
		virtual void updateWeights() override;

		virtual size_t getInputsNumber() const override;
		virtual size_t getOutputsNumber() const override;
		virtual NeuronContainer<Type> getInputs() const override;
		virtual NeuronContainer<Type> getOutputs() const override;
		virtual const float getInput(size_t index) const override;
		virtual const float getOutput(size_t index) const override;
		inline virtual AbstractLayer<AbstractBackpropagationNeuron> const* getInputLayer() const override { return m_layers.inputs(); }
		inline virtual AbstractLayer<AbstractBackpropagationNeuron> const* getOutputLayer() const override { return m_layers.outputs(); }
		virtual bool check_compatibility(AbstractNetwork const* other) const override;

		inline virtual void calculate() override { m_layers.outputs()->calculate(); }

		inline virtual void for_each_hidden(std::function<void(AbstractLayer<AbstractBackpropagationNeuron>&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_hidden(lambda, firstToLast);
		}
		inline virtual void for_each_hidden(std::function<void(AbstractLayer<AbstractBackpropagationNeuron>&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_hidden(lambda, firstToLast);
		}
		inline virtual void for_each_layer(std::function<void(AbstractLayer<AbstractBackpropagationNeuron>&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_layer(lambda, firstToLast);
		}
		inline virtual void for_each_layer(std::function<void(AbstractLayer<AbstractBackpropagationNeuron>&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_layer(lambda, firstToLast);
		}
		inline virtual void for_each_input(std::function<void(AbstractBackpropagationNeuron&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_input(lambda, firstToLast);
		}
		inline virtual void for_each_input(std::function<void(AbstractBackpropagationNeuron&)> lambda, bool const firstToLast = true) const override {
			m_layers.for_each_input(lambda, firstToLast);
		}
		inline virtual void for_each_output(std::function<void(AbstractBackpropagationNeuron&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_output(lambda, firstToLast);
		}
		inline virtual void for_each_output(std::function<void(AbstractBackpropagationNeuron&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_output(lambda, firstToLast);
		}
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractBackpropagationNeuron&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_hidden_neuron(lambda, firstToLast);
		}
		inline virtual void for_each_hidden_neuron(std::function<void(AbstractBackpropagationNeuron&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_hidden_neuron(lambda, firstToLast);
		}
		inline virtual void for_each_neuron(std::function<void(AbstractBackpropagationNeuron&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_neuron(lambda, firstToLast);
		}
		inline virtual void for_each_neuron(std::function<void(AbstractBackpropagationNeuron&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_neuron(lambda, firstToLast);
		}

		virtual std::string print() const override;
	};
}