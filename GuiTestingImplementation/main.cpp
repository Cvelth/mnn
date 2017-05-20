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
	auto *n = MNN::generateTypicalLayerNeuralNetwork(4, 2, 0, 0, MNN::ConnectionPattern::EachFromPreviousLayerWithBias,
													 [&d, &g](MNN::AbstractNeuron* n, MNN::AbstractNeuron* in) -> float {
		return d(g);
	}, 0.15f, 0.5f);

	try {
		n->calculateWithInputs({.0f, .01f, .02f, .03f});
		auto t1 = n->getOutputs();

		for (int i = 0; i < 100; i++) {
			n->learningProcess({.2f, .3f});
			n->calculateWithInputs({.0f, .01f, .02f, .03f});
		}
		auto t2 = n->getOutputs();
	} catch (MNN::Exceptions::WrongInputsNumberException e) {
		//To handle wrong input;
		QErrorMessage *t = new QErrorMessage;
		t->showMessage("Wrong input data.");
	} catch (MNN::Exceptions::WrongOutputNumberException e) {
		//To handle wrong output;
		QErrorMessage *t = new QErrorMessage;
		t->showMessage("Wrong output data.");
	}

    return a.exec();
}
