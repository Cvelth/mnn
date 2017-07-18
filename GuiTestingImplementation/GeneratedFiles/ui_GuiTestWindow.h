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
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GuiTestWindowClass
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QPushButton *generateLayerNetworkButton;
    QHBoxLayout *dataLayout;
    QHBoxLayout *inputsLayout;
    QSpacerItem *horizontalSpacer;

    void setupUi(QWidget *GuiTestWindowClass)
    {
        if (GuiTestWindowClass->objectName().isEmpty())
            GuiTestWindowClass->setObjectName(QStringLiteral("GuiTestWindowClass"));
        GuiTestWindowClass->resize(844, 591);
        verticalLayout = new QVBoxLayout(GuiTestWindowClass);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        generateLayerNetworkButton = new QPushButton(GuiTestWindowClass);
        generateLayerNetworkButton->setObjectName(QStringLiteral("generateLayerNetworkButton"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(generateLayerNetworkButton->sizePolicy().hasHeightForWidth());
        generateLayerNetworkButton->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(generateLayerNetworkButton);


        verticalLayout->addLayout(horizontalLayout);

        dataLayout = new QHBoxLayout();
        dataLayout->setSpacing(6);
        dataLayout->setObjectName(QStringLiteral("dataLayout"));
        inputsLayout = new QHBoxLayout();
        inputsLayout->setSpacing(6);
        inputsLayout->setObjectName(QStringLiteral("inputsLayout"));

        dataLayout->addLayout(inputsLayout);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        dataLayout->addItem(horizontalSpacer);

        dataLayout->setStretch(0, 1);
        dataLayout->setStretch(1, 2);

        verticalLayout->addLayout(dataLayout);

        verticalLayout->setStretch(0, 1);
        verticalLayout->setStretch(1, 5);

        retranslateUi(GuiTestWindowClass);

        QMetaObject::connectSlotsByName(GuiTestWindowClass);
    } // setupUi

    void retranslateUi(QWidget *GuiTestWindowClass)
    {
        GuiTestWindowClass->setWindowTitle(QApplication::translate("GuiTestWindowClass", "mnn Tester v0.1.dev.001", Q_NULLPTR));
        generateLayerNetworkButton->setText(QApplication::translate("GuiTestWindowClass", "Generate new BackPropaganion Layer-based Neural Network", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class GuiTestWindowClass: public Ui_GuiTestWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GUITESTWINDOW_H
