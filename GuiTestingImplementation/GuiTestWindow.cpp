#include "GuiTestWindow.hpp"
#include "MultiField.hpp"
GuiTestWindow::GuiTestWindow(QWidget *parent)
	: QWidget(parent), m_currentNetwork(nullptr) {
	ui.setupUi(this);

	m_inputs = new MultiField();
	m_outputs = new MultiField();

	ui.inputsLayout->addWidget(m_inputs);
	ui.inputsLayout->addWidget(m_outputs);

	connect(ui.generateLayerNetworkButton, &QPushButton::clicked, this, &GuiTestWindow::generateNetworkButtonSlot);
}
GuiTestWindow::~GuiTestWindow() {
	delete m_inputs;
	delete m_outputs;
	delete m_currentNetwork;
}
#include "AbstractLayerNetwork.hpp"
void GuiTestWindow::insertNetwork(mnn::AbstractLayerNetwork* network) {
	if (m_currentNetwork) delete m_currentNetwork;
	m_currentNetwork = network;
	m_inputs->change(network->getInputsNumber());
	m_outputs->change(network->getOutputsNumber());
}
#include "NetworkGenerationWindow.h"
using namespace std::placeholders;
void GuiTestWindow::generateNetworkButtonSlot() {
	auto w = new NetworkGenerationWindow(this, std::bind(&GuiTestWindow::insertNetwork, this, _1));
}