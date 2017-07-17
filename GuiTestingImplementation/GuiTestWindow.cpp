#include "GuiTestWindow.h"

GuiTestWindow::GuiTestWindow(QWidget *parent)
	: QWidget(parent) {
	ui.setupUi(this);
}

void GuiTestWindow::insertNetwork(MNN::AbstractLayerNetwork* network) {
	m_currentNetwork = network;
}

#include "NetworkGenerationWindow.h"
#include "AbstractLayerNetwork.hpp"
void generateNetworkButtonSlot() {
	std::function<void(MNN::AbstractLayerNetwork*)> f = &GuiTestWindow::insertNetwork;
	NetworkGenerationWindow* w = new NetworkGenerationWindow(f);
}