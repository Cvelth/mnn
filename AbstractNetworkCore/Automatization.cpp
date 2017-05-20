#include "Automatization.hpp"
#include "Neuron.hpp"
#include "Layer.hpp"
#include "LayerNetwork.hpp"
#include "ErrorSystem.h"

MNN::AbstractLayerNetwork* MNN::generateTypicalLayerNeuralNetwork(size_t inputs_number, size_t outputs_number, size_t hidden_layers_number, size_t neurons_per_hidden_layer, 
																  ConnectionPattern connection, std::function<float(MNN::AbstractNeuron*, MNN::AbstractNeuron*)> weightFunction, 
																  float eta, float alpha) {
	size_t i;
	MNN::AbstractLayer* in = new MNN::Layer();
	for (i = 0; i < inputs_number; i++)
		in->add(new MNN::Neuron(NeuronConstants(eta, alpha)));
	MNN::AbstractLayer* out = new MNN::Layer();
	for (i = 0; i < outputs_number; i++)
		out->add(new MNN::Neuron(NeuronConstants(eta, alpha)));

	MNN::AbstractLayerNetwork* ret = new MNN::LayerNetwork(in, out, new RootMeanSquareError());
		
	for (i = 0; i < hidden_layers_number; i++) {
		MNN::AbstractLayer* hd = new MNN::Layer();
		for (size_t j = 0; j < neurons_per_hidden_layer; j++)
			hd->add(new MNN::Neuron(NeuronConstants(eta, alpha)));
		ret->addLayer(hd);
	}

	MNN::AbstractLayer *tempLayer = in;
	switch (connection) {
		case MNN::ConnectionPattern::NoDefaultConnection:
			break;
		case MNN::ConnectionPattern::EachFromPreviousLayerWithoutBias:
			ret->for_each_hidden([&tempLayer, &weightFunction](MNN::AbstractLayer* l) {
				l->for_each([&tempLayer, &weightFunction](MNN::AbstractNeuron* n) {
					tempLayer->for_each([&n, &weightFunction](MNN::AbstractNeuron* in) {
						n->addInput(in, weightFunction(n, in));
					});
				});
				tempLayer = l;
			});
			ret->for_each_output([&tempLayer, &weightFunction](MNN::AbstractNeuron* n) {
				tempLayer->for_each([&n, &weightFunction](MNN::AbstractNeuron* in) {
					n->addInput(in, weightFunction(n, in));
				});
			});
			break;
		case MNN::ConnectionPattern::EachFromPreviousLayerWithBias:
			MNN::AbstractNeuron *bias = new MNN::Neuron(1.f);
			ret->for_each_hidden([&tempLayer, &weightFunction, &bias](MNN::AbstractLayer* layer) {
				layer->for_each([&tempLayer, &weightFunction, &bias](MNN::AbstractNeuron* neuron) {
					tempLayer->for_each([&neuron, &weightFunction](MNN::AbstractNeuron* input) {
						neuron->addInput(input, weightFunction(neuron, input));
					});
					neuron->addInput(bias, weightFunction(neuron, bias));
				});
				tempLayer = layer;
			});
			ret->for_each_output([&tempLayer, &weightFunction, &bias](MNN::AbstractNeuron* neuron) {
				tempLayer->for_each([&neuron, &weightFunction](MNN::AbstractNeuron* input) {
					neuron->addInput(input, weightFunction(neuron, input));
				});
				neuron->addInput(bias, weightFunction(neuron, bias));
			});
			break;
	}
	return ret;
}
