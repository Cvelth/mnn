#include "GuiTestWindow.h"

GuiTestWindow::GuiTestWindow(QWidget *parent)
	: QWidget(parent) {
	ui.setupUi(this);

	connect(ui.generateLayerNetworkButton, &QPushButton::clicked, this, &GuiTestWindow::generateNetworkButtonSlot);
}

void GuiTestWindow::insertNetwork(MNN::AbstractLayerNetwork* network) {
	m_currentNetwork = network;
}

#include "NetworkGenerationWindow.h"
#include "AbstractLayerNetwork.hpp"
using namespace std::placeholders;
void GuiTestWindow::generateNetworkButtonSlot() {
	auto w = new NetworkGenerationWindow(this, std::bind(&GuiTestWindow::insertNetwork, this, _1));
}