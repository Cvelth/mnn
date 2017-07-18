#pragma once

#include <QtWidgets/QWidget>
#include "ui_GuiTestWindow.h"

namespace mnn {
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
	mnn::AbstractLayerNetwork* m_currentNetwork;

	MultiField *m_inputs;
	MultiField *m_outputs;

protected slots:
	void generateNetworkButtonSlot();

public slots:
	void insertNetwork(mnn::AbstractLayerNetwork* network);
};
