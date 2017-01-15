/********************************************************************************
** Form generated from reading UI file 'GuiTestingImplementation.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GUITESTINGIMPLEMENTATION_H
#define UI_GUITESTINGIMPLEMENTATION_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GuiTestingImplementationClass
{
public:

    void setupUi(QWidget *GuiTestingImplementationClass)
    {
        if (GuiTestingImplementationClass->objectName().isEmpty())
            GuiTestingImplementationClass->setObjectName(QStringLiteral("GuiTestingImplementationClass"));
        GuiTestingImplementationClass->resize(600, 400);

        retranslateUi(GuiTestingImplementationClass);

        QMetaObject::connectSlotsByName(GuiTestingImplementationClass);
    } // setupUi

    void retranslateUi(QWidget *GuiTestingImplementationClass)
    {
        GuiTestingImplementationClass->setWindowTitle(QApplication::translate("GuiTestingImplementationClass", "GuiTestingImplementation", 0));
    } // retranslateUi

};

namespace Ui {
    class GuiTestingImplementationClass: public Ui_GuiTestingImplementationClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GUITESTINGIMPLEMENTATION_H
