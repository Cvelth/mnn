#pragma once
#include <QtWidgets\QScrollArea>
#include <QVector>
#include "Shared.hpp"
class QVBoxLayout;
class QSpacerItem;
class QDoubleSpinBox;
using Field = QDoubleSpinBox;
GenerateNewException(BrokenSizeException);
GenerateNewException(NoMoreElementsException);

class MultiField : public QScrollArea {
public:
	explicit MultiField(size_t number = 0, QWidget *parent = nullptr);
	inline ~MultiField() { destroy(); }

	inline virtual void clear() { change(0); }
	virtual void add();
	virtual void add(size_t number);
	virtual void remove();
	virtual void remove(size_t number);
	virtual void change(size_t number);

protected:
	virtual void initialize();
	virtual void destroy();
	virtual void check();
	virtual Field* newField();
	virtual void deleteField(Field* le);

private:
	size_t m_number;
	QVector<Field*> m_fields;
	QVBoxLayout *m_layout;
	QSpacerItem *m_spacer;
	QWidget *m_widget;

	static const float minimum;
	static const float maximum;
	static const float step;
};