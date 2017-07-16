#pragma once

#include <QtWidgets/QWidget>
#include "ui_NetworkGenerationWindow.h"

class NetworkGenerationWindow : public QWidget {
	Q_OBJECT

public:
	NetworkGenerationWindow(QWidget *parent = Q_NULLPTR);

private:
	Ui::NetworkGenerationWindowClass ui;
};
