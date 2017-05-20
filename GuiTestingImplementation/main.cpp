#include "GuiTestingImplementation.h"
#include <QtWidgets/QApplication>
#include <qerrormessage.h>
#include <random>
#include <initializer_list>

#include "Automatization.hpp"
#include "AbstractNeuron.hpp"
#include "AbstractLayerNetwork.hpp"
#include "Exceptions.hpp"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

	GuiTestingImplementation w;
	w.show();

	std::random_device rd;
	std::mt19937_64 g(rd());
	std::uniform_real_distribution<float> d;
	auto *n = MNN::generateTypicalLayerNeuralNetwork(8, 2, 3, 8, MNN::ConnectionPattern::EachFromPreviousLayerWithBias,
													 [&d, &g](MNN::AbstractNeuron* n, MNN::AbstractNeuron* in) -> float {
		return d(g);
	});

	try {
		n->calculateWithInputs({0, .1f, .2f, .3f, .4f, .5f, .6f, .7f});
	} catch (MNN::Exceptions::WrongInputsNumberException e) {
		//To handle wrong input;
		QErrorMessage *t = new QErrorMessage;
		t->showMessage("Wrong input.");
	}

	auto t = n->getOutputs();
    return a.exec();
}
