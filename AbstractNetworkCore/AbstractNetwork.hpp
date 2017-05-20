#pragma once

namespace std {
	initializer_list<float>;
}

namespace MNN {
	class AbstractNetwork {
	private:

	protected:

	public:
		virtual void calculate() abstract;

		virtual void newInputs(const std::initializer_list<float>& inputs, bool normalize = true) abstract;
		virtual void newInputs(size_t number, float* inputs, bool normalize = true) abstract;
		void calculateWithInputs(const std::initializer_list<float>& inputs, bool normalize = true) {
			newInputs(inputs, normalize);
			calculate();
		}
		void calculateWithInputs(size_t number, float* inputs, bool normalize = true) {
			newInputs(number, inputs, normalize);
			calculate();
		}
		void learningProcess(const std::initializer_list<float>& outputs) {
			float tempNetworkError = calculateNetworkError(outputs);
			calculateGradients(outputs);
			updateWeights();
		}
		virtual float calculateNetworkError(const std::initializer_list<float>& outputs) abstract;
		virtual void calculateGradients(const std::initializer_list<float>& outputs) abstract;
		virtual void updateWeights() abstract;

		virtual const size_t getInputsNumber() const abstract;
		virtual const size_t getOutputsNumber() const abstract;

		virtual const float* getOutputs() const abstract;
	};
}