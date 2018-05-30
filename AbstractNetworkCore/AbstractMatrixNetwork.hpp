#pragma once
#include "AbstractNetwork.hpp"
#include "AbstractNetworkControlFunctions.hpp"
namespace mnn {
	class AbstractNeuron;
	class AbstractMatrixNetwork : public AbstractNetwork, public AbstractMatrixNetworkControlFunctions<AbstractNeuron> {
		public: using AbstractNetwork::AbstractNetwork;
	};
	class AbstractBackpropagationNeuron;
	class AbstractBackpropagationMatrixNetwork : public AbstractBackpropagationNetwork, public AbstractMatrixNetworkControlFunctions<AbstractBackpropagationNeuron> {
		public: using AbstractBackpropagationNetwork::AbstractBackpropagationNetwork;
	};
}