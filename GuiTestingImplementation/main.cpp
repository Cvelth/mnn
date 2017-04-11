#include "GuiTestingImplementation.h"
#include <QtWidgets/QApplication>

#include "Neuron.hpp"
#include "Layer.hpp"
#include "LayerNetwork.hpp"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

	MNN::Neuron nI[8];
	for (int i = 0; i < 8; i++)
		nI[i].setValue(i);

	MNN::Layer input;
	for (int i = 0; i < 8; i++)
		input.add(&nI[i]);

	MNN::Neuron nO[2];
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 8; j++)
			nO[i].addInput(&nI[j]);

	MNN::Layer output;
	for (int i = 0; i < 2; i++)
		output.add(&nO[i]);

	MNN::NeuralNetwork ntw(&input, &output);

	ntw.calculate();

    GuiTestingImplementation w;
    w.show();
    return a.exec();
}
