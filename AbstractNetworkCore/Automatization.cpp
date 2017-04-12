#include "Automatization.hpp"
#include "Neuron.hpp"
#include "Layer.hpp"
#include "LayerNetwork.hpp"

MNN::AbstractLayerNetwork* MNN::generateTypicalLayerNeuralNetwork(size_t inputs_number, size_t outputs_number, size_t hidden_layers_number, size_t neurons_per_hidden_layer, ConnectionPattern connection, std::function<float(MNN::AbstractNeuron*, MNN::AbstractNeuron*)> weightFunction) {
	size_t i;
	MNN::AbstractLayer* in = new MNN::Layer();
	for (i = 0; i < inputs_number; i++)
		in->add(new MNN::Neuron());
	MNN::AbstractLayer* out = new MNN::Layer();
	for (i = 0; i < outputs_number; i++)
		out->add(new MNN::Neuron());

	MNN::AbstractLayerNetwork* ret = new MNN::LayerNetwork(in, out);
		
	for (i = 0; i < hidden_layers_number; i++) {
		MNN::AbstractLayer* hd = new MNN::Layer();
		for (size_t j = 0; j < neurons_per_hidden_layer; j++)
			hd->add(new MNN::Neuron());
		ret->addLayer(hd);
	}

	switch (connection) {
		case MNN::ConnectionPattern::NoDefaultConnection:
			break;
		case MNN::ConnectionPattern::EachFromPreviousLayerWithBias:
			MNN::AbstractNeuron *bias = new MNN::Neuron(1.f);
			MNN::AbstractLayer *temp = in;
			ret->for_each_hidden([&temp, &weightFunction, &bias](MNN::AbstractLayer* layer) {
				layer->for_each([&temp, &weightFunction, &bias](MNN::AbstractNeuron* neuron) {
					temp->for_each([&neuron, &weightFunction](MNN::AbstractNeuron* input) {
						neuron->addInput(input, weightFunction(neuron, input));
					});
					neuron->addInput(bias, weightFunction(neuron, bias));
				});
				temp = layer;
			});
			ret->for_each_output([&temp, &weightFunction, &bias](MNN::AbstractNeuron* neuron) {
				temp->for_each([&neuron, &weightFunction](MNN::AbstractNeuron* input) {
					neuron->addInput(input, weightFunction(neuron, input));
				});
				neuron->addInput(bias, weightFunction(neuron, bias));
			});
			break;
		case MNN::ConnectionPattern::EachFromPreviousLayerWithoutBias:
			MNN::AbstractLayer *temp = in;
			ret->for_each_hidden([&temp, &weightFunction](MNN::AbstractLayer* l) {
				l->for_each([&temp, &weightFunction](MNN::AbstractNeuron* n) {
					temp->for_each([&n, &weightFunction](MNN::AbstractNeuron* in) {
						n->addInput(in, weightFunction(n, in));
					});
				});
				temp = l;
			});
			ret->for_each_output([&temp, &weightFunction](MNN::AbstractNeuron* n) {
				temp->for_each([&n, &weightFunction](MNN::AbstractNeuron* in) {
					n->addInput(in, weightFunction(n, in));
				});
			});
			break;
	}
	return ret;
}
