#pragma once
#include "AbstractNetwork.hpp"
#include "AbstractNetworkControlFunctions.hpp"
namespace mnn {
	class AbstractNeuron;
	class AbstractMatrixNetwork : public virtual AbstractNetwork, public virtual AbstractMatrixNetworkControlFunctions<AbstractNeuron> {
		public: using AbstractNetwork::AbstractNetwork;
	};
	class AbstractBackpropagationNeuron;
	class AbstractBackpropagationMatrixNetwork : public virtual AbstractBackpropagationNetwork, public virtual AbstractMatrixNetworkControlFunctions<AbstractBackpropagationNeuron> {
		public: using AbstractBackpropagationNetwork::AbstractBackpropagationNetwork;
	};
}