/********************************************************************************
** Form generated from reading UI file 'NetworkGenerationWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_NETWORKGENERATIONWINDOW_H
#define UI_NETWORKGENERATIONWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_NetworkGenerationWindowClass
{
public:

    void setupUi(QWidget *NetworkGenerationWindowClass)
    {
        if (NetworkGenerationWindowClass->objectName().isEmpty())
            NetworkGenerationWindowClass->setObjectName(QStringLiteral("NetworkGenerationWindowClass"));
        NetworkGenerationWindowClass->resize(600, 400);

        retranslateUi(NetworkGenerationWindowClass);

        QMetaObject::connectSlotsByName(NetworkGenerationWindowClass);
    } // setupUi

    void retranslateUi(QWidget *NetworkGenerationWindowClass)
    {
        NetworkGenerationWindowClass->setWindowTitle(QApplication::translate("NetworkGenerationWindowClass", "Network Generation", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class NetworkGenerationWindowClass: public Ui_NetworkGenerationWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_NETWORKGENERATIONWINDOW_H
