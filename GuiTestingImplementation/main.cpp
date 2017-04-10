#include "GuiTestingImplementation.h"
#include <QtWidgets/QApplication>

#include "Neuron.hpp"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

	MNN::Neuron n;
	n.setInputs({new MNN::Neuron(1.f), new MNN::Neuron(2.f)});
	float p = n.value();

    GuiTestingImplementation w;
    w.show();
    return a.exec();
}
