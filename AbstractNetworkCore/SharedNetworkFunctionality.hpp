#pragma once
#include "AbstractNetworkStorage.hpp"
#include "AbstractLayerNetwork.hpp"
#include "AbstractMatrixNetwork.hpp"
#include "AbstractLayer.hpp"
namespace mnn {
	template <typename NeuronType, typename StorageType>
	class SharedAbstractNetworkFunctionality : public virtual AbstractMatrixNetworkControlFunctions<NeuronType>, public virtual AbstractNetwork {
	protected:
		StorageType m_storage;
	public:
		explicit SharedAbstractNetworkFunctionality(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs) : m_storage(inputs, outputs) {}
		
		virtual void setInputs(NeuronContainer<Type> const& inputs, bool normalize = true) override {
			if (inputs.size() != m_storage.inputs()->size())
				throw Exceptions::IncorrectDataAmountException();

			auto it = inputs.begin();
			if (normalize) m_storage.inputs()->for_each([&it](NeuronType& n) { n.setValue(*(it++)); });
			else m_storage.inputs()->for_each([&it](NeuronType& n) { n.setValueUnnormalized(*(it++)); });
		}
		virtual NeuronContainer<Type> getInputs() const override {
			NeuronContainer<Type> res;
			m_storage.inputs()->for_each([&res](NeuronType& n) { res.push_back(n.value()); });
			return res;
		}
		virtual NeuronContainer<Type> getOutputs() const override {
			NeuronContainer<Type> res;
			m_storage.outputs()->for_each([&res](NeuronType& n) { res.push_back(n.value()); });
			return res;
		}
		virtual size_t getInputsNumber() const override { return getInputLayer()->size(); }
		virtual size_t getOutputsNumber() const override { return getOutputLayer()->size(); }
		virtual const float getInput(size_t index) const override { return getInputLayer()->at(index); }
		virtual const float getOutput(size_t index) const override { return getOutputLayer()->at(index); }
		virtual AbstractLayer<NeuronType> const* getInputLayer() const override { return m_storage.inputs(); }
		virtual AbstractLayer<NeuronType> const* getOutputLayer() const override { return m_storage.outputs(); }
		virtual bool check_compatibility(AbstractNetwork const* other) const override {
			auto o = dynamic_cast<StaredAbstractNetworkFunctionality const*>(other);
			if (getInputsNumber() != o->getInputsNumber())
				return false;
			if (getOutputsNumber() != o->getOutputsNumber())
				return false;
			if (m_storage->size() != o->m_storage->size())
				return false;
			return true;
		}

		virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_storage.for_each_input(lambda, firstToLast);
		}
		virtual void for_each_input(std::function<void(NeuronType&)> lambda, bool const firstToLast = true) const override {
			m_storage.for_each_input(lambda, firstToLast);
		}
		virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_storage.for_each_output(lambda, firstToLast);
		}
		virtual void for_each_output(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			m_storage.for_each_output(lambda, firstToLast);
		}
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_storage.for_each_hidden_neuron(lambda, firstToLast);
		}
		virtual void for_each_hidden_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			m_storage.for_each_hidden_neuron(lambda, firstToLast);
		}
		virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) override {
			m_storage.for_each_neuron(lambda, firstToLast);
		}
		virtual void for_each_neuron(std::function<void(NeuronType&)> lambda, bool firstToLast = true) const override {
			m_storage.for_each_neuron(lambda, firstToLast);
		}
	};
	template <typename NeuronType>
	class SharedMatrixNetworkFunctionality : public virtual SharedAbstractNetworkFunctionality<NeuronType, MatrixNetworkStorage<NeuronType>>, public virtual AbstractMatrixNetwork {
	public:
		explicit SharedMatrixNetworkFunctionality(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs) : SharedAbstractNetworkFunctionality(inputs, outputs) {}
		
		virtual void calculate() override {
			// To be implemented.
		}
	};
	template <typename NeuronType>
	class SharedLayerNetworkFunctionality : public virtual SharedAbstractNetworkFunctionality<NeuronType, LayerNetworkStorage<NeuronType>>, public virtual AbstractLayerNetworkControlFunctions<NeuronType>, public virtual AbstractLayerNetwork {
	public:
		explicit SharedLayerNetworkFunctionality(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs) : SharedAbstractNetworkFunctionality(inputs, outputs) {}
		explicit SharedLayerNetworkFunctionality(AbstractLayer<NeuronType>* inputs, AbstractLayer<NeuronType>* outputs, LayerContainer<AbstractLayer<NeuronType>*> const& c)
			: SharedLayerNetworkFunctionality(inputs, outputs) { addHiddenLayers(c); }

		virtual void addHiddenLayer(AbstractLayer<NeuronType>* l) { m_layers->push_back(l); }
		virtual void addHiddenLayers(LayerContainer<AbstractLayer<NeuronType>*> const& c) { for (auto t : c) addHiddenLayer(t); }

		virtual void calculate() override { m_layers.outputs()->calculate(); }

		virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) override {
			m_storage.for_each_hidden(lambda, firstToLast);
		}
		virtual void for_each_hidden(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const override {
			m_storage.for_each_hidden(lambda, firstToLast);
		}
		virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) override {
			m_storage.for_each_layer(lambda, firstToLast);
		}
		virtual void for_each_layer(std::function<void(AbstractLayer<NeuronType>&)> lambda, bool firstToLast = true) const override {
			m_storage.for_each_layer(lambda, firstToLast);
		}
	};
}