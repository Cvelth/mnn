#pragma once
#include "AbstractNetwork.hpp"
#include "AbstractNetworkControlFunctions.hpp"
namespace mnn {
	class AbstractNeuron;
	class AbstractLayerNetwork : public virtual AbstractNetwork/*, public virtual AbstractLayerNetworkControlFunctions<AbstractNeuron>*/ {
		public: using AbstractNetwork::AbstractNetwork;
	};
	class AbstractBackpropagationNeuron;
	class AbstractBackpropagationLayerNetwork : public virtual AbstractBackpropagationNetwork/*, public virtual AbstractLayerNetworkControlFunctions<AbstractBackpropagationNeuron>*/ {
		public: using AbstractBackpropagationNetwork::AbstractBackpropagationNetwork;
	};
}