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

		virtual const float* getOutputs() const abstract;
	};
}