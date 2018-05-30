#pragma once
#include "AbstractMatrixNetwork.hpp"
#include "Matrix.hpp"
namespace mnn {
	class MatrixNetwork : public AbstractMatrixNetwork {
	protected:
		Matrix<AbstractNeuron> m_matrix;
	public:
		//explicit 
	};
}