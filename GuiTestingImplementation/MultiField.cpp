#include "MultiField.hpp"

const float MultiField::minimum = -1.f;
const float MultiField::maximum = +1.f;
const float MultiField::step = 0.05f;

MultiField::MultiField(size_t number, QWidget * parent) : QScrollArea(parent), m_number(number) {
	initialize();
	add(number);
}

MultiField::~MultiField() {
	destroy();
}

void MultiField::clear() {
	change(0);
}

void MultiField::add() {
	m_fields.push_back(newField());
	m_number++;
}

void MultiField::add(size_t number) {
	check();
	for (int i = 0; i < number; i++)
		add();
}

void MultiField::remove() {
	if (!m_number)
		throw Exceptions::MultiField::NoMoreElementsException();

	deleteField(m_fields.back());
	m_fields.pop_back();
	m_number--;
}

void MultiField::remove(size_t number) {
	check();
	for (int i = 0; i < number; i++)
		remove();
}

void MultiField::change(size_t number) {
	check();
	signed int difference = signed int(number) - m_number;
	if (difference >= 0)
		add(difference);
	else
		remove(-difference);
}

#include <QtWidgets\QSpacerItem>
#include <QtWidgets\QVBoxLayout>
void MultiField::initialize() {
	m_layout = new QVBoxLayout();
	m_spacer = new QSpacerItem(6, 6, QSizePolicy::Expanding, QSizePolicy::Expanding);

	m_layout->addItem(m_spacer);
	this->setLayout(m_layout);
}

void MultiField::destroy() {
	if (m_number != 0)
		clear();
	m_layout->removeItem(m_spacer);
	delete m_spacer;
	delete m_layout;
}

void MultiField::check() {
	if (m_number != m_fields.size())
		throw Exceptions::MultiField::BrokenSizeException();
}

#include <QtWidgets\QDoubleSpinBox>
Field* MultiField::newField() {
	Field* ret = new Field();
	ret->setMinimum(minimum);
	ret->setMaximum(maximum);
	ret->setSingleStep(step);

	m_layout->removeItem(m_spacer);
	m_layout->addWidget(ret);
	m_layout->addItem(m_spacer);

	return ret;
}

void MultiField::deleteField(Field* le) {
	m_layout->removeWidget(le);
	delete le;
}
