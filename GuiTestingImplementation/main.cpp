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
#include "qpushbutton.h"
#include "qlayout.h"
#include "qspinbox.h"
#include "LambdaTest.hpp"
#include "Automatization.hpp"
bool b(float f) {
	return fabs(f) > 0.5f ? 1.f : 0.f;
}
QString calculate_action(size_t size) {
	mnnt::LambdaTest test([](auto& inputs, auto& outputs) {
		inputs[0] = b(inputs[0]);
		inputs[1] = b(inputs[1]);
		outputs[0] = bool(inputs[0]) ^ bool(inputs[1]);
	});

	test.insertNeuralNetwork(mnn::generateTypicalLayerNeuralNetwork(2, 1, 5, 5,
		mnn::ConnectionPattern::EachFromPreviousLayerWithBias,
		mnn::random_weights, 0.15f, 0.5f));
	test.calculate();

	QString output;
	bool in[2], o2;
	float o1;
	for (int i = 0; i < size; i++) {
		in[0] = b(test.getInput(0));
		in[1] = b(test.getInput(1));
		o1 = test.getOutput(0);
		o2 = in[0] ^ in[1];
		test.learningProcess();
		output += QString::number(in[0]) + "  ^  " + QString::number(in[1]) + "  =  " +
			QString::number(o2) + "    " + QString::number(b(o1)) + "  ->  " +
			(b(o1) == o2 ? "*" : " ") + "  " + QString::number(o1) + '\n';
	}
	return output;
}
int main(int argc, char *argv[]) {
	QApplication a(argc, argv);
	QWidget w;
	QHBoxLayout *ul = new QHBoxLayout();
	QSpinBox s;
	s.setRange(1, std::numeric_limits<int>::max());
	s.setValue(2000);
	QFont font("Consolas", 15);
	s.setFont(font);
	ul->addWidget(&s);
	QPushButton b("Calculate");
	ul->addWidget(&b);
	QVBoxLayout *l = new QVBoxLayout();
	l->addLayout(ul);
	QTextBrowser t;
	t.setFont(font);
	l->addWidget(&t);
	auto action = [&t, &s]() { t.setText(calculate_action(s.value())); };
	QObject::connect(&b, &QPushButton::clicked, action);
	action();
	w.setLayout(l);
	w.setMinimumWidth(600);
	w.show();
	return a.exec();
}
/**/