#include "GuiTestingImplementation.h"
#include <QtWidgets/QApplication>

#include "Neuron.hpp"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

	MNN::Neuron n;
	n.setInputs({1,2,3,4});
	float p = n.value();

    GuiTestingImplementation w;
    w.show();
    return a.exec();
}
