#include "Automatization.hpp"
#include "Neuron.hpp"
#include "Layer.hpp"
#include "LayerNetwork.hpp"
mnn::AbstractLayerNetwork* mnn::generateTypicalLayerNeuralNetwork(size_t inputs_number, size_t outputs_number, size_t hidden_layers_number,
																  size_t neurons_per_hidden_layer, ConnectionPattern connection,
																  std::function<Type(AbstractNeuron const&, AbstractNeuron const&)> weightFunction) {
	size_t i;
	mnn::AbstractLayer<AbstractNeuron>* in = new mnn::Layer<AbstractNeuron>();
	for (i = 0; i < inputs_number; i++)
		in->add(new mnn::Neuron());
	mnn::AbstractLayer<AbstractNeuron>* out = new mnn::Layer<AbstractNeuron>();
	for (i = 0; i < outputs_number; i++)
		out->add(new mnn::Neuron());

	mnn::AbstractLayerNetwork* ret = new mnn::LayerNetwork(in, out);

	for (i = 0; i < hidden_layers_number; i++) {
		mnn::AbstractLayer<AbstractNeuron>* hd = new mnn::Layer<AbstractNeuron>();
		for (size_t j = 0; j < neurons_per_hidden_layer; j++)
			hd->add(new mnn::Neuron());
		ret->addHiddenLayer(hd);
	}

	mnn::AbstractLayer<AbstractNeuron> *tempLayer = in;
	switch (connection) {
		case mnn::ConnectionPattern::NoDefaultConnection:
			break;
		case mnn::ConnectionPattern::EachFromPreviousLayerWithoutBias:
			ret->for_each_hidden([&tempLayer, &weightFunction](mnn::AbstractLayer<AbstractNeuron>& l) {
				l.for_each([&tempLayer, &weightFunction](mnn::AbstractNeuron& n) {
					tempLayer->for_each([&n, &weightFunction](mnn::AbstractNeuron& in) {
						n.link(&in, weightFunction(n, in));
					});
				});
				tempLayer = &l;
			});
			ret->for_each_output([&tempLayer, &weightFunction](mnn::AbstractNeuron& n) {
				tempLayer->for_each([&n, &weightFunction](mnn::AbstractNeuron& in) {
					n.link(&in, weightFunction(n, in));
				});
			});
			break;
		case mnn::ConnectionPattern::EachFromPreviousLayerWithBias:
			mnn::AbstractNeuron *bias = new mnn::Neuron(1.f);
			ret->for_each_hidden([&tempLayer, &weightFunction, &bias](mnn::AbstractLayer<AbstractNeuron>& layer) {
				layer.for_each([&tempLayer, &weightFunction, &bias](mnn::AbstractNeuron& neuron) {
					tempLayer->for_each([&neuron, &weightFunction](mnn::AbstractNeuron& input) {
						neuron.link(&input, weightFunction(neuron, input));
					});
					neuron.link(bias, weightFunction(neuron, *bias));
				});
				tempLayer = &layer;
			});
			ret->for_each_output([&tempLayer, &weightFunction, &bias](mnn::AbstractNeuron& neuron) {
				tempLayer->for_each([&neuron, &weightFunction](mnn::AbstractNeuron& input) {
					neuron.link(&input, weightFunction(neuron, input));
				});
				neuron.link(bias, weightFunction(neuron, *bias));
			});
			break;
	}
	return ret;
}

mnn::AbstractBackpropagationLayerNetwork* mnn::generateTypicalBackpropagationLayerNeuralNetwork(size_t inputs_number, size_t outputs_number, size_t hidden_layers_number,
																								size_t neurons_per_hidden_layer, ConnectionPattern connection,
																								std::function<Type(AbstractNeuron const&, AbstractNeuron const&)> weightFunction,
																								Type eta, Type alpha) {
	size_t i;
	mnn::AbstractLayer<AbstractBackpropagationNeuron>* in = new mnn::Layer<AbstractBackpropagationNeuron>();
	for (i = 0; i < inputs_number; i++)
		in->add(new mnn::BackpropagationNeuron(eta, alpha));
	mnn::AbstractLayer<AbstractBackpropagationNeuron>* out = new mnn::Layer<AbstractBackpropagationNeuron>();
	for (i = 0; i < outputs_number; i++)
		out->add(new mnn::BackpropagationNeuron(eta, alpha));

	mnn::AbstractBackpropagationLayerNetwork* ret = new mnn::BackpropagationLayerNetwork(in, out);

	for (i = 0; i < hidden_layers_number; i++) {
		mnn::AbstractLayer<AbstractBackpropagationNeuron>* hd = new mnn::Layer<AbstractBackpropagationNeuron>();
		for (size_t j = 0; j < neurons_per_hidden_layer; j++)
			hd->add(new mnn::BackpropagationNeuron(eta, alpha));
		ret->addHiddenLayer(hd);
	}

	mnn::AbstractLayer<AbstractBackpropagationNeuron> *tempLayer = in;
	switch (connection) {
		case mnn::ConnectionPattern::NoDefaultConnection:
			break;
		case mnn::ConnectionPattern::EachFromPreviousLayerWithoutBias:
			ret->for_each_hidden([&tempLayer, &weightFunction](mnn::AbstractLayer<AbstractBackpropagationNeuron>& l) {
				l.for_each([&tempLayer, &weightFunction](mnn::AbstractBackpropagationNeuron& n) {
					tempLayer->for_each([&n, &weightFunction](mnn::AbstractBackpropagationNeuron& in) {
						n.link(&in, weightFunction(n, in));
					});
				});
				tempLayer = &l;
			});
			ret->for_each_output([&tempLayer, &weightFunction](mnn::AbstractBackpropagationNeuron& n) {
				tempLayer->for_each([&n, &weightFunction](mnn::AbstractBackpropagationNeuron& in) {
					n.link(&in, weightFunction(n, in));
				});
			});
			break;
		case mnn::ConnectionPattern::EachFromPreviousLayerWithBias:
			mnn::AbstractBackpropagationNeuron *bias = new mnn::BackpropagationNeuron(1.f, eta, alpha);
			ret->for_each_hidden([&tempLayer, &weightFunction, &bias](mnn::AbstractLayer<AbstractBackpropagationNeuron>& layer) {
				layer.for_each([&tempLayer, &weightFunction, &bias](mnn::AbstractBackpropagationNeuron& neuron) {
					tempLayer->for_each([&neuron, &weightFunction](mnn::AbstractBackpropagationNeuron& input) {
						neuron.link(&input, weightFunction(neuron, input));
					});
					neuron.link(bias, weightFunction(neuron, *bias));
				});
				tempLayer = &layer;
			});
			ret->for_each_output([&tempLayer, &weightFunction, &bias](mnn::AbstractBackpropagationNeuron& neuron) {
				tempLayer->for_each([&neuron, &weightFunction](mnn::AbstractBackpropagationNeuron& input) {
					neuron.link(&input, weightFunction(neuron, input));
				});
				neuron.link(bias, weightFunction(neuron, *bias));
			});
			break;
	}
	return ret;
}
float mnn::default_weights(AbstractNeuron const& neuron, AbstractNeuron const& input) {
	return 1.0f;
}
#include <random>
std::mt19937_64 g((std::random_device())());
std::uniform_real_distribution<float> d(-1.f, +1.f);
std::bernoulli_distribution chance(0.5);
float mnn::random_weights(AbstractNeuron const& neuron, AbstractNeuron const& input) {
	return d(g);
}
#include "Shared.hpp"
mnn::AbstractLayerNetwork* mnn::generateTypicalLayerNeuralNetwork(AbstractLayerNetwork *parent1, AbstractLayerNetwork *parent2) {
	if (!parent1 || !parent2)
		throw Exceptions::UnsupportedParameters();
	if (!parent1->check_compatibility(parent2))
		throw Exceptions::UnsupportedParameters();

	size_t i;
	mnn::AbstractLayer<AbstractNeuron>* in = new mnn::Layer<AbstractNeuron>();
	for (i = 0; i < parent1->getInputsNumber(); i++)
		in->add(new mnn::Neuron());
	mnn::AbstractLayer<AbstractNeuron>* out = new mnn::Layer<AbstractNeuron>();
	for (i = 0; i < parent1->getOutputsNumber(); i++)
		out->add(new mnn::Neuron());

	mnn::AbstractLayerNetwork* ret = new mnn::LayerNetwork(in, out);

	parent1->for_each_hidden([ret](mnn::AbstractLayer<AbstractNeuron> const& l) {
		mnn::AbstractLayer<AbstractNeuron>* hd = new mnn::Layer<AbstractNeuron>();
		l.for_each([hd](AbstractNeuron const& n) {
			hd->add(new mnn::Neuron());
		});
		ret->addHiddenLayer(hd);
	});

	mnn::AbstractLayer<AbstractNeuron> *tempLayer = in;
	//Supports only mnn::ConnectionPattern::EachFromPreviousLayerWithoutBias.
	//Other are to be added in future updates.
	NeuronContainer<Type> c;
	parent1->for_each_neuron([&c](mnn::AbstractNeuron const& n) {
		n.for_each_link([&c](mnn::Link const& l) {
			c.push_back(l.weight);
		});
	});
	i = 0u;
	parent2->for_each_neuron([&c, &i](mnn::AbstractNeuron const& n) {
		n.for_each_link([&c, &i](mnn::Link const& l) {
			if (chance(g))
				c[i] = l.weight;
			i++;
		});
	});

	i = 0u;
	mnn::AbstractNeuron *bias = new mnn::Neuron(1.f);
	ret->for_each_hidden([&tempLayer, &c, &i, &bias](mnn::AbstractLayer<AbstractNeuron>& layer) {
		layer.for_each([&tempLayer, &c, &i, &bias](mnn::AbstractNeuron& neuron) {
			tempLayer->for_each([&neuron, &c, &i](mnn::AbstractNeuron& input) {
				neuron.link(&input, c[i++]);
			});
			neuron.link(bias, c[i++]);
		});
		tempLayer = &layer;
	});
	ret->for_each_output([&tempLayer, &c, &i, &bias](mnn::AbstractNeuron& neuron) {
		tempLayer->for_each([&neuron, &c, &i](mnn::AbstractNeuron& input) {
			neuron.link(&input, c[i++]);
		});
		neuron.link(bias, c[i++]);
	});
	return ret;
}