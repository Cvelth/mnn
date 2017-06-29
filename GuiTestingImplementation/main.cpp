#include "GuiTestingImplementation.h"
#include <QtWidgets/QApplication>
#include <qerrormessage.h>

#include "StaticDataTest.hpp"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

	MNNT::AbstractTest *test;

	test = new MNNT::StaticDataTest({ .0f, .01f, .02f, .03f }, { .2f, .3f });

	test->generateNeuralNetwork();
	test->calculate();

	float o1[100], o2[100];
	for (int i = 0; i < 100; i++) {
		o1[i] = test->getOutput(0);
		o2[i] = test->getOutput(1);
		test->learningProcess();
	}

	delete test;

	GuiTestingImplementation w;
	w.show();
    return a.exec();
}
