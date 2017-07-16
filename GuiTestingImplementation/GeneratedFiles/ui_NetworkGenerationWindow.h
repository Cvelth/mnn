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
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_NetworkGenerationWindowClass
{
public:
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QSpinBox *inputs;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_3;
    QSpinBox *outputs;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_4;
    QSpinBox *hidden;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_5;
    QSpinBox *per_hidden;
    QComboBox *connection;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_6;
    QComboBox *default_weight;
    QPushButton *moreButton;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_7;
    QDoubleSpinBox *eta;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_8;
    QDoubleSpinBox *alpha;
    QSpacerItem *verticalSpacer_2;
    QPushButton *pushButton;

    void setupUi(QWidget *NetworkGenerationWindowClass)
    {
        if (NetworkGenerationWindowClass->objectName().isEmpty())
            NetworkGenerationWindowClass->setObjectName(QStringLiteral("NetworkGenerationWindowClass"));
        NetworkGenerationWindowClass->resize(483, 513);
        verticalLayout = new QVBoxLayout(NetworkGenerationWindowClass);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        label = new QLabel(NetworkGenerationWindowClass);
        label->setObjectName(QStringLiteral("label"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);
        QFont font;
        font.setPointSize(16);
        label->setFont(font);
        label->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Minimum);

        verticalLayout->addItem(verticalSpacer);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label_2 = new QLabel(NetworkGenerationWindowClass);
        label_2->setObjectName(QStringLiteral("label_2"));

        horizontalLayout->addWidget(label_2);

        inputs = new QSpinBox(NetworkGenerationWindowClass);
        inputs->setObjectName(QStringLiteral("inputs"));
        inputs->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        inputs->setMinimum(1);
        inputs->setValue(2);

        horizontalLayout->addWidget(inputs);

        horizontalLayout->setStretch(1, 1);

        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        label_3 = new QLabel(NetworkGenerationWindowClass);
        label_3->setObjectName(QStringLiteral("label_3"));

        horizontalLayout_2->addWidget(label_3);

        outputs = new QSpinBox(NetworkGenerationWindowClass);
        outputs->setObjectName(QStringLiteral("outputs"));
        outputs->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        outputs->setMinimum(1);

        horizontalLayout_2->addWidget(outputs);

        horizontalLayout_2->setStretch(1, 1);

        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        label_4 = new QLabel(NetworkGenerationWindowClass);
        label_4->setObjectName(QStringLiteral("label_4"));

        horizontalLayout_3->addWidget(label_4);

        hidden = new QSpinBox(NetworkGenerationWindowClass);
        hidden->setObjectName(QStringLiteral("hidden"));
        hidden->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_3->addWidget(hidden);

        horizontalLayout_3->setStretch(1, 1);

        verticalLayout->addLayout(horizontalLayout_3);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        label_5 = new QLabel(NetworkGenerationWindowClass);
        label_5->setObjectName(QStringLiteral("label_5"));

        horizontalLayout_4->addWidget(label_5);

        per_hidden = new QSpinBox(NetworkGenerationWindowClass);
        per_hidden->setObjectName(QStringLiteral("per_hidden"));
        per_hidden->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_4->addWidget(per_hidden);

        horizontalLayout_4->setStretch(1, 1);

        verticalLayout->addLayout(horizontalLayout_4);

        connection = new QComboBox(NetworkGenerationWindowClass);
        connection->setObjectName(QStringLiteral("connection"));

        verticalLayout->addWidget(connection);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        label_6 = new QLabel(NetworkGenerationWindowClass);
        label_6->setObjectName(QStringLiteral("label_6"));

        horizontalLayout_5->addWidget(label_6);

        default_weight = new QComboBox(NetworkGenerationWindowClass);
        default_weight->setObjectName(QStringLiteral("default_weight"));

        horizontalLayout_5->addWidget(default_weight);

        horizontalLayout_5->setStretch(1, 1);

        verticalLayout->addLayout(horizontalLayout_5);

        moreButton = new QPushButton(NetworkGenerationWindowClass);
        moreButton->setObjectName(QStringLiteral("moreButton"));

        verticalLayout->addWidget(moreButton);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        label_7 = new QLabel(NetworkGenerationWindowClass);
        label_7->setObjectName(QStringLiteral("label_7"));

        horizontalLayout_6->addWidget(label_7);

        eta = new QDoubleSpinBox(NetworkGenerationWindowClass);
        eta->setObjectName(QStringLiteral("eta"));
        eta->setMaximum(1);
        eta->setSingleStep(0.05);
        eta->setValue(0.15);

        horizontalLayout_6->addWidget(eta);

        horizontalLayout_6->setStretch(1, 1);

        verticalLayout->addLayout(horizontalLayout_6);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        label_8 = new QLabel(NetworkGenerationWindowClass);
        label_8->setObjectName(QStringLiteral("label_8"));

        horizontalLayout_7->addWidget(label_8);

        alpha = new QDoubleSpinBox(NetworkGenerationWindowClass);
        alpha->setObjectName(QStringLiteral("alpha"));
        alpha->setMaximum(1);
        alpha->setSingleStep(0.05);
        alpha->setValue(0.5);

        horizontalLayout_7->addWidget(alpha);

        horizontalLayout_7->setStretch(1, 1);

        verticalLayout->addLayout(horizontalLayout_7);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Minimum);

        verticalLayout->addItem(verticalSpacer_2);

        pushButton = new QPushButton(NetworkGenerationWindowClass);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(pushButton->sizePolicy().hasHeightForWidth());
        pushButton->setSizePolicy(sizePolicy1);

        verticalLayout->addWidget(pushButton);


        retranslateUi(NetworkGenerationWindowClass);

        QMetaObject::connectSlotsByName(NetworkGenerationWindowClass);
    } // setupUi

    void retranslateUi(QWidget *NetworkGenerationWindowClass)
    {
        NetworkGenerationWindowClass->setWindowTitle(QApplication::translate("NetworkGenerationWindowClass", "Network Generation", Q_NULLPTR));
        label->setText(QApplication::translate("NetworkGenerationWindowClass", "Insert the data to be used in \n"
"new network generation:", Q_NULLPTR));
        label_2->setText(QApplication::translate("NetworkGenerationWindowClass", "Input neurons number:", Q_NULLPTR));
        label_3->setText(QApplication::translate("NetworkGenerationWindowClass", "Output neurons number:", Q_NULLPTR));
        label_4->setText(QApplication::translate("NetworkGenerationWindowClass", "Hidden layers number:", Q_NULLPTR));
        label_5->setText(QApplication::translate("NetworkGenerationWindowClass", "Neurons per a hidden layer:", Q_NULLPTR));
        connection->clear();
        connection->insertItems(0, QStringList()
         << QApplication::translate("NetworkGenerationWindowClass", "Connect to each neuron from previous layer", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "Connect to each neuron from previous layer including bias neuron", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "No connections (Not recommended)", Q_NULLPTR)
        );
        label_6->setText(QApplication::translate("NetworkGenerationWindowClass", "Default weight function:", Q_NULLPTR));
        default_weight->clear();
        default_weight->insertItems(0, QStringList()
         << QApplication::translate("NetworkGenerationWindowClass", "random (-1.f, 1.f) (recommended)", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "random (0.f, 1.f)", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "random (-1.f, 0.f)", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "1.f", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "0.5f", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "0.f (Not recommended)", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "-0.5f", Q_NULLPTR)
         << QApplication::translate("NetworkGenerationWindowClass", "-1.f", Q_NULLPTR)
        );
        moreButton->setText(QApplication::translate("NetworkGenerationWindowClass", "More", Q_NULLPTR));
        label_7->setText(QApplication::translate("NetworkGenerationWindowClass", "eta:", Q_NULLPTR));
        label_8->setText(QApplication::translate("NetworkGenerationWindowClass", "alpha:", Q_NULLPTR));
        pushButton->setText(QApplication::translate("NetworkGenerationWindowClass", "Generate", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class NetworkGenerationWindowClass: public Ui_NetworkGenerationWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_NETWORKGENERATIONWINDOW_H
