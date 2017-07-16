#include <QtWidgets/QApplication>
#include "qtextbrowser.h"

#include "LogicalFunctionTest.hpp"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

	MNNT::LogicalFunctionTest test(MNNT::LogicalFunction::ExOr);

	test.generateNeuralNetwork();
	test.calculate();

	const size_t ITERATIONS = 8000;

	QString output;

	float i1[ITERATIONS], i2[ITERATIONS];
	float o1[ITERATIONS], o2[ITERATIONS];
	float sum1 = 0.f, sum2 = 0.f;
	for (int i = 0; i < ITERATIONS; i++) {
		i1[i] = test.getInput(0);
		sum1 += i1[i];
		i2[i] = test.getInput(1);
		sum2 += i2[i];
		o1[i] = test.getOutput(0);
		o2[i] = float(bool(i1[i]) ^ bool(i2[i]));
		test.learningProcess();
		output += QString::number(i1[i]) + "   " + QString::number(i2[i]) + '\t' + QString::number(o2[i]) + "   " + (fabs(o1[i]) > 0.5f ? '1' : '0') + "   " + QString::number(o1[i]) + '\n';
	}

	QTextBrowser w;
	w.setText(output);
	w.show();
    return a.exec();
}
