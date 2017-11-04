#include <QtWidgets\QApplication>
/*
#include "GuiTestWindow.h"
int main(int argc, char **argv) {
QApplication app(argc, argv);
GuiTestWindow w;
w.show();
return app.exec();
}
/*/
#include "qtextbrowser.h"
#include "LambdaTest.hpp"
#include "Automatization.hpp"
bool b(float f) {
	return fabs(f) > 0.5f ? 1.f : 0.f;
}
int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	mnnt::LambdaTest test([](auto& inputs, auto& outputs) {
		inputs[0] = b(inputs[0]);
		inputs[1] = b(inputs[1]);
		outputs[0] = bool(inputs[0]) ^ bool(inputs[1]);
	});

	test.insertNeuralNetwork(mnn::generateTypicalLayerNeuralNetwork(2, 1, 1, 3,
		mnn::ConnectionPattern::EachFromPreviousLayerWithBias,
		mnn::random_weights, 0.15f, 0.5f));
	test.calculate();

	QString output;
	float in[2], o;
	for (int i = 0; i < 2000; i++) {
		in[0] = test.getInput(0);
		in[1] = test.getInput(1);
		o = test.getOutput(0);
		test.learningProcess();
		output += QString::number(b(in[0])) + "   " + QString::number(b(in[1])) + '\t'+ 
			QString::number(float(b(in[0]) ^ b(in[1]))) + "   " + 
			QString::number(b(o)) +  + "   " + QString::number(o) + '\n';
	}

	QTextBrowser w;
	w.setText(output);
	w.show();
	return a.exec();
}
/**/