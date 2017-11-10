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
			return 0.f;
	}, 2, 1);
	
	em->changeSelectionParameters(0.8);
	em->newPopulation();
	for (size_t i = 0; i < 5; i++) {
		em->testPopulation(true);
		em->populationSelection();
		em->recreatePopulation();
		em->mutatePopulation(1.f, 1.f);
	}

	return 0;// a.exec();
}
/**/