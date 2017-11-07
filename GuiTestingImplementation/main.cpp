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
#include "NetworkGenerationEvolutionManager.hpp"
int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	auto em = new mnn::NetworkGenerationEvolutionManager(100, [](auto calculate) -> Type {
		if (calculate({0,0})[0] > 0.f)
			return 1.f;
		else
			return -1.f;
	}, 2, 1);

	em->changeSelectionParameters(0.2, mnn::SelectionType::Value);
	em->nextGeneration();
	em->testPopulation();

	return a.exec();
}
/**/