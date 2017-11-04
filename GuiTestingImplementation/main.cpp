#include <QtWidgets\QApplication>
//*
#include "GuiTestWindow.hpp"
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
#include "qscrollbar.h"
#include "LambdaTest.hpp"
#include "Automatization.hpp"
bool b(float f) {
	return fabs(f) > 0.5f ? 1.f : 0.f;
}
QString calculate_action(size_t size, bool all_outputs_included = false) {
	mnnt::LambdaTest test([](auto& inputs, auto& outputs) {
		inputs[0] = b(inputs[0]);
		inputs[1] = b(inputs[1]);
		outputs[0] = bool(inputs[0]) ^ bool(inputs[1]);
	});

	test.insertNeuralNetwork(mnn::generateTypicalLayerNeuralNetwork(2, 1, 5, 6,
		mnn::ConnectionPattern::EachFromPreviousLayerWithBias,
		mnn::random_weights, 0.15f, 0.5f));
	test.calculate();

	QString output;
	size_t last_error = 0;
	for (int i = 0; i < size && i - last_error < 5000; i++) {
		test.learningProcess();
		if (all_outputs_included) {
			bool e, in[2];
			in[0] = b(test.getInput(0));
			in[1] = b(test.getInput(1));
			float o1 = test.getOutput(0);
			bool o2 = in[0] ^ in[1];
			if (!(e = b(o1) == o2)) last_error = i;
			output += QString::number(in[0]) + "  ^  " + QString::number(in[1]) + "  =  " +
				QString::number(o2) + "  " + QString::number(b(o1)) + "  ->  " +
				(e ? "*" : " ") + "  " + QString::number(o1) + "\t\t" + (e ? QString::number(i) : " ") + '\n';
		} else
			if (b(test.getOutput(0)) != (b(test.getInput(0)) ^ b(test.getInput(1)))) last_error = i;
	}
	return (all_outputs_included ? output : "") + "Last error: " + QString::number(last_error);
}
QString calculate_multiple_actions(size_t size, size_t number) {
	QString output;
	for (int i = 0; i < number; i++)
		output += calculate_action(size, false) + '\n';
	return output;
}
int main(int argc, char *argv[]) {
	QApplication a(argc, argv);
	QWidget w;
	QHBoxLayout *ul = new QHBoxLayout();
	QSpinBox s;
	s.setRange(1, std::numeric_limits<int>::max());
	s.setValue(15000);
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
	auto action = [&t, &s]() { 
		//t.setText(calculate_action(s.value(), true));
		t.setText(calculate_multiple_actions(s.value(), 40));
		t.verticalScrollBar()->setValue(t.verticalScrollBar()->maximum());
	};
	QObject::connect(&b, &QPushButton::clicked, action);
	action();
	w.setLayout(l);
	w.showMaximized();
	return a.exec();
}
/**/