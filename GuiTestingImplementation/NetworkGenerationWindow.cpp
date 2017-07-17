#include "NetworkGenerationWindow.h"

NetworkGenerationWindow::NetworkGenerationWindow(QObject* receiver, void(QObject::*slot)(), QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);

	hideAdditionalFields();

	connect(ui.moreButton, &QPushButton::clicked, this, &NetworkGenerationWindow::toggleAdditionalFields);
	connect(ui.generateButton, &QPushButton::clicked, this, &NetworkGenerationWindow::startGeneration);
	connect(this, &NetworkGenerationWindow::returnNetwork, receiver, slot);

	show();
}

void NetworkGenerationWindow::hideAdditionalFields() {
	ui.eta_label->hide();
	ui.eta->hide();
	//ui.eta_layout->hide();

	ui.alpha_label->hide();
	ui.alpha->hide();
	//ui.alpha_layout->hide();

	ui.moreButton->setText("More");
	areAdditionalFieldsShown = false;
}

void NetworkGenerationWindow::showAdditionalFields() {
	ui.eta_label->show();
	ui.eta->show();
	//ui.eta_layout->show();

	ui.alpha_label->show();
	ui.alpha->show();
	//ui.alpha_layout->show();

	ui.moreButton->setText("Less");
	areAdditionalFieldsShown = true;
}

void NetworkGenerationWindow::toggleAdditionalFields() {
	if (areAdditionalFieldsShown)
		hideAdditionalFields();
	else
		showAdditionalFields();
}

#include "Automatization.hpp"
void NetworkGenerationWindow::startGeneration() {
	auto network = MNN::generateTypicalLayerNeuralNetwork(ui.inputs->value(), ui.outputs->value(), ui.hidden->value(), ui.per_hidden->value(),
														  chooseConnection(ui.connection->currentIndex()), chooseDefaultWeights(ui.default_weight->currentIndex()),
														  ui.eta->value(), ui.alpha->value());
	emit returnNetwork(network);
}

MNN::ConnectionPattern NetworkGenerationWindow::chooseConnection(size_t index) {
	switch (index) {
		default:
		case 0:
			return MNN::ConnectionPattern::EachFromPreviousLayerWithBias;
		case 1:
			return MNN::ConnectionPattern::EachFromPreviousLayerWithoutBias;
		case 2:
			return MNN::ConnectionPattern::NoDefaultConnection;
	}
}
#include "AbstractNeuron.hpp"
#include <random>
std::function<float(MNN::AbstractNeuron*, MNN::AbstractNeuron*)> NetworkGenerationWindow::chooseDefaultWeights(size_t index) {
	std::mt19937_64 g = std::random_device()();
	switch (index) {
		default:
		case 0: //Random (-1.f, 1.f)
			std::uniform_real_distribution<> d(-1.f, +1.f);
			return [&g, &d](MNN::AbstractNeuron* neuron, MNN::AbstractNeuron* input) -> float {
				return d(g);
			};
		case 1: //Random (-0.f, 1.f)
			std::uniform_real_distribution<> d(-0.f, +1.f);
			return [&g, &d](MNN::AbstractNeuron* neuron, MNN::AbstractNeuron* input) -> float {
				return d(g);
			};
		case 2: //Random (-1.f, 0.f)
			std::uniform_real_distribution<> d(-1.f, +0.f);
			return [&g, &d](MNN::AbstractNeuron* neuron, MNN::AbstractNeuron* input) -> float {
				return d(g);
			};
		case 3: //Always +1.f
			return [](MNN::AbstractNeuron* neuron, MNN::AbstractNeuron* input) -> float {
				return +1.f;
			};
		case 4: //Always +0.5f
			return [](MNN::AbstractNeuron* neuron, MNN::AbstractNeuron* input) -> float {
				return +0.5f;
			};
		case 5: //Always +0.f
			return [](MNN::AbstractNeuron* neuron, MNN::AbstractNeuron* input) -> float {
				return +0.f;
			};
		case 6: //Always -0.5f
			return [](MNN::AbstractNeuron* neuron, MNN::AbstractNeuron* input) -> float {
				return -0.5f;
			};
		case 7: //Always -1.f
			return [](MNN::AbstractNeuron* neuron, MNN::AbstractNeuron* input) -> float {
				return -1.f;
			};
	}
	return std::function<float(MNN::AbstractNeuron*, MNN::AbstractNeuron*)>();
}
