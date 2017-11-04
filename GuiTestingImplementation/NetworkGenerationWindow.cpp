#include "NetworkGenerationWindow.hpp"
bool NetworkGenerationWindow::isGeneratorInitialized = false;
mnnt::RealRandomEngine* NetworkGenerationWindow::m_random_generator = nullptr;
NetworkGenerationWindow::~NetworkGenerationWindow() {
	if (isGeneratorInitialized) delete m_random_generator;
	isGeneratorInitialized = false;
}
void NetworkGenerationWindow::hideAdditionalFields() {
	ui.eta_label->hide();
	ui.eta->hide();

	ui.alpha_label->hide();
	ui.alpha->hide();

	ui.moreButton->setText("More");
	areAdditionalFieldsShown = false;
}
void NetworkGenerationWindow::showAdditionalFields() {
	ui.eta_label->show();
	ui.eta->show();

	ui.alpha_label->show();
	ui.alpha->show();

	ui.moreButton->setText("Less");
	areAdditionalFieldsShown = true;
}
void NetworkGenerationWindow::toggleAdditionalFields() {
	if (areAdditionalFieldsShown)
		hideAdditionalFields();
	else
		showAdditionalFields();
}
#include <AbstractNetwork.hpp>
NetworkGenerationWindow::NetworkGenerationWindow(QObject* receiver, std::function<void(mnn::AbstractNetwork*)> slot, QWidget *parent)
	: QWidget(parent) {
	ui.setupUi(this);

	hideAdditionalFields();
	connect(ui.moreButton, &QPushButton::clicked, this, &NetworkGenerationWindow::toggleAdditionalFields);
	connect(ui.generateButton, &QPushButton::clicked, this, &NetworkGenerationWindow::startGeneration);
	connect(this, &NetworkGenerationWindow::returnNetwork, receiver, slot);
	show();
}
#include "AbstractNeuron.hpp"
#include "RandomEngine.hpp"
std::function<float(mnn::AbstractNeuron const&, mnn::AbstractNeuron const&)> NetworkGenerationWindow::chooseDefaultWeights(size_t index) {
	if (!isGeneratorInitialized) {
		m_random_generator = new mnnt::RealRandomEngine();
		isGeneratorInitialized = true;
	}
	auto generatorPointer = m_random_generator;
	switch (index) {
		default:
		case 0: //Random (-1.f, 1.f)
			if (!isGeneratorInitialized)
				m_random_generator->changeDistribution(-1.f, +1.f);
			return [generatorPointer](mnn::AbstractNeuron const& neuron, mnn::AbstractNeuron const& input) -> float {
				return (*generatorPointer)();
			};
		case 1: //Random (-0.f, 1.f)
			if (!isGeneratorInitialized)
				m_random_generator->changeDistribution(-0.f, +1.f);
			return [generatorPointer](mnn::AbstractNeuron const& neuron, mnn::AbstractNeuron const& input) -> float {
				return (*generatorPointer)();
			};
		case 2: //Random (-1.f, 0.f)
			if (!isGeneratorInitialized)
				m_random_generator->changeDistribution(-1.f, +0.f);
			return [generatorPointer](mnn::AbstractNeuron const& neuron, mnn::AbstractNeuron const& input) -> float {
				return (*generatorPointer)();
			};
		case 3: //Always +1.f
			return [](mnn::AbstractNeuron const& neuron, mnn::AbstractNeuron const& input) -> float {
				return +1.f;
			};
		case 4: //Always +0.5f
			return [](mnn::AbstractNeuron const& neuron, mnn::AbstractNeuron const& input) -> float {
				return +0.5f;
			};
		case 5: //Always +0.f
			return [](mnn::AbstractNeuron const& neuron, mnn::AbstractNeuron const& input) -> float {
				return +0.f;
			};
		case 6: //Always -0.5f
			return [](mnn::AbstractNeuron const& neuron, mnn::AbstractNeuron const& input) -> float {
				return -0.5f;
			};
		case 7: //Always -1.f
			return [](mnn::AbstractNeuron const& neuron, mnn::AbstractNeuron const& input) -> float {
				return -1.f;
			};
	}
	return std::function<float(mnn::AbstractNeuron const&, mnn::AbstractNeuron const&)>();
}
#include "Automatization.hpp"
mnn::ConnectionPattern NetworkGenerationWindow::chooseConnection(size_t index) {
	switch (index) {
	default:
	case 0:
		return mnn::ConnectionPattern::EachFromPreviousLayerWithBias;
	case 1:
		return mnn::ConnectionPattern::EachFromPreviousLayerWithoutBias;
	case 2:
		return mnn::ConnectionPattern::NoDefaultConnection;
	}
}
#include "AbstractLayerNetwork.hpp"
void NetworkGenerationWindow::startGeneration() {
	mnn::AbstractNetwork *network = mnn::generateTypicalLayerNeuralNetwork(ui.inputs->value(), ui.outputs->value(), ui.hidden->value(), ui.per_hidden->value(),
		chooseConnection(ui.connection->currentIndex()), chooseDefaultWeights(ui.default_weight->currentIndex()),
		ui.eta->value(), ui.alpha->value());
	emit returnNetwork(network);
	delete this;
}