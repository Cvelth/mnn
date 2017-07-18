#pragma once

#include <QtWidgets/QWidget>
#include "ui_GuiTestWindow.h"

namespace MNN {
	class AbstractLayerNetwork;
}
class MultiField;

class GuiTestWindow : public QWidget {
	Q_OBJECT

public:
	GuiTestWindow(QWidget *parent = Q_NULLPTR);
	~GuiTestWindow();

private:
	Ui::GuiTestWindowClass ui;
	MNN::AbstractLayerNetwork* m_currentNetwork;

	MultiField *m_inputs;
	MultiField *m_outputs;

protected slots:
	void generateNetworkButtonSlot();

public slots:
	void insertNetwork(MNN::AbstractLayerNetwork* network);
};
