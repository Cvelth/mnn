#pragma once

#include <QtWidgets/QWidget>
#include "ui_GuiTestingImplementation.h"

class GuiTestingImplementation : public QWidget
{
    Q_OBJECT

public:
    GuiTestingImplementation(QWidget *parent = Q_NULLPTR);

private:
    Ui::GuiTestingImplementationClass ui;
};
