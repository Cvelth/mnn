#include "Automatization.hpp"
#include "Neuron.hpp"
#include "Layer.hpp"
#include "LayerNetwork.hpp"
#include "ErrorSystem.h"

mnn::AbstractLayerNetwork* mnn::generateTypicalLayerNeuralNetwork(size_t inputs_number, size_t outputs_number, size_t hidden_layers_number, size_t neurons_per_hidden_layer,
																  ConnectionPattern connection, std::function<float(mnn::AbstractNeuron*, mnn::AbstractNeuron*)> weightFunction, 
																  float eta, float alpha) {
	size_t i;
	mnn::AbstractLayer* in = new mnn::Layer();
	for (i = 0; i < inputs_number; i++)
		in->add(new mnn::Neuron(NeuronConstants(eta, alpha)));
	mnn::AbstractLayer* out = new mnn::Layer();
	for (i = 0; i < outputs_number; i++)
		out->add(new mnn::Neuron(NeuronConstants(eta, alpha)));

	mnn::AbstractLayerNetwork* ret = new mnn::LayerNetwork(in, out, new ErrorSystems::MeanSquareError());
		
	for (i = 0; i < hidden_layers_number; i++) {
		mnn::AbstractLayer* hd = new mnn::Layer();
		for (size_t j = 0; j < neurons_per_hidden_layer; j++)
			hd->add(new mnn::Neuron(NeuronConstants(eta, alpha)));
		ret->addLayer(hd);
	}

	mnn::AbstractLayer *tempLayer = in;
	switch (connection) {
		case mnn::ConnectionPattern::NoDefaultConnection:
			break;
		case mnn::ConnectionPattern::EachFromPreviousLayerWithoutBias:
			ret->for_each_hidden([&tempLayer, &weightFunction](mnn::AbstractLayer* l) {
				l->for_each([&tempLayer, &weightFunction](mnn::AbstractNeuron* n) {
					tempLayer->for_each([&n, &weightFunction](mnn::AbstractNeuron* in) {
						n->addInput(in, weightFunction(n, in));
					});
				});
				tempLayer = l;
			});
			ret->for_each_output([&tempLayer, &weightFunction](mnn::AbstractNeuron* n) {
				tempLayer->for_each([&n, &weightFunction](mnn::AbstractNeuron* in) {
					n->addInput(in, weightFunction(n, in));
				});
			});
			break;
		case mnn::ConnectionPattern::EachFromPreviousLayerWithBias:
			mnn::AbstractNeuron *bias = new mnn::Neuron(1.f);
			ret->for_each_hidden([&tempLayer, &weightFunction, &bias](mnn::AbstractLayer* layer) {
				layer->for_each([&tempLayer, &weightFunction, &bias](mnn::AbstractNeuron* neuron) {
					tempLayer->for_each([&neuron, &weightFunction](mnn::AbstractNeuron* input) {
						neuron->addInput(input, weightFunction(neuron, input));
					});
					neuron->addInput(bias, weightFunction(neuron, bias));
				});
				tempLayer = layer;
			});
			ret->for_each_output([&tempLayer, &weightFunction, &bias](mnn::AbstractNeuron* neuron) {
				tempLayer->for_each([&neuron, &weightFunction](mnn::AbstractNeuron* input) {
					neuron->addInput(input, weightFunction(neuron, input));
				});
				neuron->addInput(bias, weightFunction(neuron, bias));
			});
			break;
	}
	return ret;
}

float mnn::default_weights(AbstractNeuron *neuron, AbstractNeuron *input) {
	return 1.0f;
}
#include <random>
std::mt19937_64 g((std::random_device())());
std::uniform_real_distribution<float> d(-1.f, +1.f);
float mnn::random_weights(AbstractNeuron *neuron, AbstractNeuron *input) {
	return d(g);
}