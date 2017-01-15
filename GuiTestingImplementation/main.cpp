#include "GuiTestingImplementation.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GuiTestingImplementation w;
    w.show();
    return a.exec();
}
