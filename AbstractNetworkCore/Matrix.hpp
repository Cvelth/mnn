#pragma once
namespace mnn {
	template <typename UnitType>
	class Matrix {
	public:
		void for_each(std::function<void(UnitType&)> lambda, bool firstToLast = true) {
			//To be implemented.
		}
		void for_each(std::function<void(UnitType&)> lambda, bool firstToLast = true) const {
			//To be implemented.
		}
	};
}