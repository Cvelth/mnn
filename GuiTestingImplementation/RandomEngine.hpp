#pragma once
#include <random>

namespace MNNT {
	template <class RandomEngine, class Distribution, typename ReturnType>
	class AbstractRandomEngine {
	private:
		RandomEngine* m_engine;
		Distribution* m_distribution;
	public:
		explicit AbstractRandomEngine() {
			m_engine = new RandomEngine(std::random_device()());
			m_distribution = new Distribution();
		}

		~AbstractRandomEngine() {
			delete m_engine;
			delete m_distribution;
		}

		ReturnType operator()() {
			return (ReturnType) (*m_distribution)(*m_engine);
		}
	};

	using RealRandomEngine = AbstractRandomEngine<std::mt19937_64, std::uniform_real_distribution<float>, float>;
	using BinaryRandomEngine = AbstractRandomEngine<std::mt19937_64, std::discrete_distribution<float>, float>;
}