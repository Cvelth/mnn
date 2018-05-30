#pragma once
#include "Shared.hpp"
#include "AbstractLayerNetwork.hpp"
#include "LayerNetworkStorage.hpp"
#include "AbstractNeuron.hpp"
namespace mnn {
	template <typename NeuronType>
	class SharedLayerNetworkFunctionality : public virtual AbstractNetwork, public virtual AbstractLayerNetworkControlFunctions<NeuronType> {
	protected:
		LayerNetworkStorage<NeuronType> m_layers;
	public:
		explicit SharedLayerNetworkFunctionality(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs) : m_layers(inputs, outputs) {}
		explicit SharedLayerNetworkFunctionality(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs, LayerContainer<AbstractLayer<NeuronType>*> const& c)
			: SharedLayerNetworkFunctionality(inputs, outputs) { addHiddenLayers(c); }

		virtual void addHiddenLayer(AbstractLayer<NeuronType>* l) { m_layers->push_back(l); }
		virtual void addHiddenLayers(LayerContainer<AbstractLayer<NeuronType>*> const& c) { for (auto t : c) addHiddenLayer(t); }

		virtual void setInputs(NeuronContainer<Type> const& inputs, bool normalize = true) override {
			if (inputs.size() != m_layers.inputs()->size())
				throw Exceptions::IncorrectDataAmountException();

			auto it = inputs.begin();
			if (normalize) m_layers.inputs()->for_each([&it](NeuronType& n) { n.setValue(*(it++)); });
			else m_layers.inputs()->for_each([&it](NeuronType& n) { n.setValueUnnormalized(*(it++)); });
		}
		virtual NeuronContainer<Type> getInputs() const override {
			NeuronContainer<Type> res;
			m_layers.inputs()->for_each([&res](NeuronType& n) { res.push_back(n.value()); });
			return res;
		}
		virtual NeuronContainer<Type> getOutputs() const override {
			NeuronContainer<Type> res;
			m_layers.outputs()->for_each([&res](NeuronType& n) { res.push_back(n.value()); });
			return res;
		}
		virtual size_t getInputsNumber() const override { return getInputLayer()->size(); }
		virtual size_t getOutputsNumber() const override { return getOutputLayer()->size();	}
		virtual const float getInput(size_t index) const override { return getInputLayer()->at(index); }
		virtual const float getOutput(size_t index) const override { return getOutputLayer()->at(index); }
		virtual AbstractLayer<NeuronType> const* getInputLayer() const override { return m_layers.inputs(); }
		virtual AbstractLayer<NeuronType> const* getOutputLayer() const override { return m_layers.outputs(); }
		virtual bool check_compatibility(AbstractNetwork const* other) const override {
			auto o = dynamic_cast<SharedLayerNetworkFunctionality const*>(other);
			if (getInputsNumber() != o->getInputsNumber())
				return false;
			if (getOutputsNumber() != o->getOutputsNumber())
				return false;
			if (m_layers->size() != o->m_layers->size())
				return false;
			return true;
		}

		virtual void calculate() override { m_layers.outputs()->calculate(); }

		virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_hidden(lambda, firstToLast);
		}
		virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_hidden(lambda, firstToLast);
		}
		virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_layer(lambda, firstToLast);
		}
		virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_layer(lambda, firstToLast);
		}
		virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_input(lambda, firstToLast);
		}
		virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool const firstToLast = true) const override {
			m_layers.for_each_input(lambda, firstToLast);
		}
		virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_output(lambda, firstToLast);
		}
		virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_output(lambda, firstToLast);
		}
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_hidden_neuron(lambda, firstToLast);
		}
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_hidden_neuron(lambda, firstToLast);
		}
		virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_layers.for_each_neuron(lambda, firstToLast);
		}
		virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			m_layers.for_each_neuron(lambda, firstToLast);
		}
	};
	class LayerNetwork : public virtual AbstractLayerNetwork, public virtual SharedLayerNetworkFunctionality<AbstractNeuron> {
	public:
		using SharedLayerNetworkFunctionality::SharedLayerNetworkFunctionality;
		virtual std::string print() const override;
	};

	class BackpropagationLayerNetwork : public virtual AbstractBackpropagationLayerNetwork, public virtual SharedLayerNetworkFunctionality<AbstractBackpropagationNeuron> {
	public:
		using SharedLayerNetworkFunctionality::SharedLayerNetworkFunctionality;
		virtual std::string print() const override;

		void calculateGradients(const NeuronContainer<Type>& outputs) override {
			if (outputs.size() != m_layers.outputs()->size())
				throw Exceptions::IncorrectDataAmountException();

			auto it = outputs.begin();
			SharedLayerNetworkFunctionality<AbstractBackpropagationNeuron>::for_each_output([&it](AbstractBackpropagationNeuron& n) { n.calculateGradient(*(it++)); });

			AbstractLayer<AbstractBackpropagationNeuron>* nextLayer = m_layers.outputs();
			for_each_hidden([&nextLayer](AbstractLayer<AbstractBackpropagationNeuron>& l) {
				l.for_each([&nextLayer](AbstractBackpropagationNeuron& n) {
					n.calculateGradient([&nextLayer](std::function<Type(AbstractBackpropagationNeuron&)> calculate_unit) -> Type {
						Type sum = Type(0.f);
						nextLayer->for_each([&sum, &calculate_unit](AbstractBackpropagationNeuron& nn) {
							sum += calculate_unit(nn);
						});
						return sum;
					});
				});
				nextLayer = &l;
			}, false);
		}
		virtual void updateWeights() override {
			for_each_neuron([](AbstractBackpropagationNeuron& n) {
				n.recalculateWeights();
			}, false);
		}
	};
}