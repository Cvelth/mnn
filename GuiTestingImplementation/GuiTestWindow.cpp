#include "GuiTestWindow.hpp"
#include "MultiField.hpp"
#include "AbstractNetwork.hpp"
#include "NetworkGenerationWindow.hpp"
#include <QFileDialog>
GuiTestWindow::GuiTestWindow(QWidget *parent)
	: QWidget(parent), m_currentNetwork(nullptr) {
	ui.setupUi(this);

	m_inputs_layout = new QVBoxLayout;
	m_inputs_layout->addWidget(m_inputs_field = new MultiField());
	m_inputs_layout->addWidget(m_calculate = new QPushButton("Calculate ->"));
	m_outputs_layout = new QVBoxLayout;
	m_outputs_layout->addWidget(m_outputs_field = new MultiField());
	m_outputs_layout->addWidget(m_learn = new QPushButton("<- Learn"));

	ui.inputsLayout->addLayout(m_inputs_layout);
	ui.inputsLayout->addLayout(m_outputs_layout);

	connect(m_calculate, &QPushButton::clicked, this, &GuiTestWindow::calculate);
	connect(m_learn, &QPushButton::clicked, this, &GuiTestWindow::learn);
	connect(ui.save, &QPushButton::clicked, this, &GuiTestWindow::save);
	connect(ui.load, &QPushButton::clicked, this, &GuiTestWindow::load);
	connect(ui.generateLayerNetworkButton, &QPushButton::clicked, [this]() {
		auto w = new NetworkGenerationWindow(this, 
			std::bind(&GuiTestWindow::insertNetwork, this, std::placeholders::_1));
		w->setAttribute(Qt::WA_DeleteOnClose);
	});
}
GuiTestWindow::~GuiTestWindow() {
	delete m_currentNetwork;
	delete m_inputs_field;
	delete m_calculate;
	delete m_outputs_field;
	delete m_learn;
}
void GuiTestWindow::insertNetwork(mnn::AbstractBackpropagationNetwork* network) {
	if (m_currentNetwork) delete m_currentNetwork;
	m_currentNetwork = network;
	if (m_currentNetwork) {
		m_inputs_field->change(network->getInputsNumber());
		m_outputs_field->change(network->getOutputsNumber());
	} else
		throw Exceptions::NoNeuralNetworkInserted();
}
void GuiTestWindow::calculate() {
	if (!m_currentNetwork)
		throw Exceptions::NoNeuralNetworkInserted();
	m_currentNetwork->calculate();
	auto results = m_currentNetwork->getOutputs();
	if (results.size() != m_outputs_field->size())
		throw Exceptions::NeuralNetworkSupportError();
	for (size_t i = 0; i < results.size(); i++)
		m_outputs_field->at(i)->setValue(results[i]);
}
void GuiTestWindow::learn() {
	if (!m_currentNetwork)
		throw Exceptions::NoNeuralNetworkInserted();
	if (m_currentNetwork->getOutputsNumber() != m_outputs_field->size())
		throw Exceptions::NeuralNetworkSupportError();
	auto outputs = NeuronContainer<Type>(m_outputs_field->size());
	for (size_t i = 0; i < outputs.size(); i++)
		outputs[i] = m_outputs_field->at(i)->value();
	m_currentNetwork->learningProcess(outputs);
}
#include "Storage.hpp"
void GuiTestWindow::save() {
	if (!m_currentNetwork)
		throw Exceptions::NoNeuralNetworkInserted();
	mnn::save_to_file(QFileDialog::getSaveFileName(this, tr("Save Neural Network"), "", 
												   tr("MyNeuralNetwork file (*.mnn)")).toStdString(), 
		              m_currentNetwork);
}
void GuiTestWindow::load() {
	insertNetwork(dynamic_cast<mnn::AbstractBackpropagationNetwork*>(mnn::load_from_file(
		QFileDialog::getOpenFileName(this, tr("Open Neural Network"), "",
									 tr("MyNeuralNetwork file (*.mnn)")).toStdString())));
}