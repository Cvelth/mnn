#include "GuiTestingImplementation.h"
#include <QtWidgets/QApplication>

#include "Automatization.hpp"
#include "AbstractNeuron.hpp"
#include "AbstractLayerNetwork.hpp"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

	auto *n = MNN::generateTypicalLayerNeuralNetwork(8, 2, 0, 0, MNN::ConnectionPattern::EachFromPreviousLayerWithBias);

	float f = 0.f;
	n->for_each_input([&f](MNN::AbstractNeuron* n) {
		n->setValue(f += 0.1f);
	});
	n->calculate();

    GuiTestingImplementation w;
    w.show();
    return a.exec();
}
