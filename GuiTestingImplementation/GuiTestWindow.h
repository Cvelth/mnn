#pragma once

#include <QtWidgets/QWidget>
#include "ui_GuiTestWindow.h"

namespace MNN {
	class AbstractLayerNetwork;
}

class GuiTestWindow : public QWidget {
	Q_OBJECT

public:
	GuiTestWindow(QWidget *parent = Q_NULLPTR);

private:
	Ui::GuiTestWindowClass ui;
	MNN::AbstractLayerNetwork* m_currentNetwork;

protected slots:
	void generateNetworkButtonSlot();

public slots:
	void insertNetwork(MNN::AbstractLayerNetwork* network);
};
