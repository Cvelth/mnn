#include <QtWidgets\QApplication>
/*
#include "GuiTestWindow.hpp"
int main(int argc, char **argv) {
	QApplication app(argc, argv);
	GuiTestWindow w;
	w.show();
	return app.exec();
}
/*/
#include "GenerationEvolutionManager.hpp"
int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	auto em = new mnn::GenerationEvolutionManager(100, [](auto calculate) -> Type {
		if (calculate({0,0})[0] > 0.f)
			return 1.f;
		else
			return 0.f;
	});

	return a.exec();
}
//using EvaluationFunction = std::function<Type(std::function<NeuronContainer<Type>(NeuronContainer<Type>)>)>;
/**/