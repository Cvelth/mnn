#pragma once
#include "AbstractTest.hpp"
#include <vector>

namespace mnn {
	class AbstractLayerNetwork;
}

namespace mnnt {
	class AbstractStaticTest : public AbstractTest {
	protected:
		mnn::AbstractLayerNetwork* m_network;
	public:
		AbstractStaticTest() : AbstractTest() {};
		~AbstractStaticTest();

		virtual void generateNeuralNetwork(size_t inputs, size_t outputs, size_t hidden, size_t per_hidden) override;
		virtual const size_t getOutputsNumber() const override;
		virtual const float* getOutputs() const override;
		virtual const float getOutput(size_t index) const override;
	};

	/*
	Neural Network test realization allowing to check the stability of the Network.
	During learning it feeds the network with the same list of static input data which is unchangable during object existence
	The learning process is executed of the output data array.
	Both inputs and outputs are passed in the Constructor.
	
	All the methods inherit base class's, For more details see AbstractTest.hpps
	*/
	class StaticDataTest : public AbstractStaticTest {
	protected:
		std::initializer_list<float> m_inputs;
		std::initializer_list<float> m_outputs;
	public:
		/*
		Constructor of static test class.
		Parameters:
		* static_inputs - the data passed into Network in every iteration.
		* static_output - expected output data for the input.
		*/
		StaticDataTest(const std::initializer_list<float>& static_inputs, const std::initializer_list<float>& static_outputs) 
						: AbstractStaticTest(), m_inputs(static_inputs), m_outputs(static_outputs) {}

		virtual void generateNeuralNetwork();

		virtual void calculate() override;
		virtual void learningProcess() override;
	};

	namespace Exceptions {
		class NoDataException {};
		class EmptyDataException {};
		class IncorrectDataSizeException {};
	}

	class StaticMultiDataTest : public AbstractStaticTest {
		protected:
			std::vector<std::initializer_list<float>> m_inputs;
			std::vector<std::initializer_list<float>> m_outputs;
			size_t m_current_index;
		protected:
			virtual void incrementIndex();
			virtual bool checkData() const;
		public:
			//Constructor of static test class.
			StaticMultiDataTest() : AbstractStaticTest() {}

			virtual void addDataSet(const std::initializer_list<float>& inputs, const std::initializer_list<float>& outputs);
			virtual void generateNeuralNetwork();

			virtual void calculate() override;
			virtual void learningProcess() override;
	};
}