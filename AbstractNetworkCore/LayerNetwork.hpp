#pragma once
#include "NetworkContainer.hpp"
#include "AbstractLayerNetwork.hpp"
#include "AbstractLayer.hpp"

namespace MNN {
	class AbstractErrorSystem;
}

namespace MNN {
	class LayerNetwork : public AbstractLayerNetwork {
	private:

	protected:
		NetworkDataContainer<AbstractLayer*> m_layers;
		AbstractLayer* m_inputs;
		AbstractLayer* m_outputs;
		AbstractErrorSystem* m_errorSystem;
	public:
		explicit LayerNetwork(AbstractLayer* inputs, AbstractLayer* outputs, AbstractErrorSystem* errorSystem) : m_inputs(inputs), m_outputs(outputs), m_errorSystem(errorSystem) {}
		inline virtual void addLayer(AbstractLayer* l) override {
			m_layers.push_back(l);
		}
		inline virtual void addLayers(const NetworkDataContainer<AbstractLayer*>& c) {
			for (auto t : c)
				this->addLayer(t);
		}
		explicit LayerNetwork(AbstractLayer* inputs, AbstractLayer* outputs, AbstractErrorSystem* errorSystem, const NetworkDataContainer<AbstractLayer*>& c) 
							: LayerNetwork(inputs, outputs, errorSystem) {
			this->addLayers(c);
		}

		virtual void newInputs(const std::initializer_list<float>& inputs, bool normalize = true) override;
		virtual void newInputs(size_t number, float* inputs, bool normalize = true) override;
		virtual void newInputs(const NetworkDataContainer<float>& inputs, bool normalize = true);

		void calculateWithInputs(const NetworkDataContainer<float>& inputs, bool normalize = true);
		void learningProcess(const NetworkDataContainer<float>& outputs);

		virtual const size_t getInputsNumber() const override {
			return m_inputs->size();
		}
		virtual const size_t getOutputsNumber() const override {
			return m_outputs->size();
		}

		virtual const float* getOutputs() const override;
		
		inline virtual void calculate() override {
			m_outputs->calculate();
		}

		inline virtual void for_each_hidden(std::function<void(AbstractLayer*)> lambda, bool firstToLast = true) override {
			for (auto it = firstToLast ? m_layers.begin() : m_layers.end(); 
				 it != (firstToLast ? m_layers.end() : m_layers.begin()); 
				 firstToLast ? it++ : it--) 
			{
				lambda(*it);
			}
		}
		inline virtual void for_each(std::function<void(AbstractLayer*)> lambda, bool firstToLast = true) override {
			if (firstToLast) lambda(m_inputs);
			else lambda(m_outputs);

			for_each_hidden(lambda, firstToLast);

			if (firstToLast) lambda(m_outputs);
			else lambda(m_inputs);
		}
		inline virtual void for_each_input(std::function<void(AbstractNeuron*)> lambda, bool firstToLast = true) override {
			m_inputs->for_each(lambda, firstToLast);
		}
		inline virtual void for_each_output(std::function<void(AbstractNeuron*)> lambda, bool firstToLast = true) override {
			m_outputs->for_each(lambda, firstToLast);
		}
		inline virtual void for_each_neuron(std::function<void(AbstractNeuron*)> lambda, bool firstToLast = true) override {
			if (firstToLast) m_inputs->for_each(lambda, firstToLast);
			else m_outputs->for_each(lambda, firstToLast);

			for (auto it = firstToLast ? m_layers.begin() : m_layers.end();
				 it != (firstToLast ? m_layers.end() : m_layers.begin());
				 firstToLast ? it++ : it--) {
				(*it)->for_each(lambda, firstToLast);
			}

			if (firstToLast) m_outputs->for_each(lambda, firstToLast);
			else m_inputs->for_each(lambda, firstToLast);
		}
	};
}