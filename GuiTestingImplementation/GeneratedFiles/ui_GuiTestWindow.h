/********************************************************************************
** Form generated from reading UI file 'GuiTestWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GUITESTWINDOW_H
#define UI_GUITESTWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GuiTestWindowClass
{
public:

    void setupUi(QWidget *GuiTestWindowClass)
    {
        if (GuiTestWindowClass->objectName().isEmpty())
            GuiTestWindowClass->setObjectName(QStringLiteral("GuiTestWindowClass"));
        GuiTestWindowClass->resize(600, 400);

        retranslateUi(GuiTestWindowClass);

        QMetaObject::connectSlotsByName(GuiTestWindowClass);
    } // setupUi

    void retranslateUi(QWidget *GuiTestWindowClass)
    {
        GuiTestWindowClass->setWindowTitle(QApplication::translate("GuiTestWindowClass", "MNN Tester v0.1.dev.001", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class GuiTestWindowClass: public Ui_GuiTestWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GUITESTWINDOW_H
