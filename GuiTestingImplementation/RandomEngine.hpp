#pragma once
#include <random>

namespace mnnt {
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
			if (m_engine) delete m_engine;
			if (m_distribution) delete m_distribution;
		}

		void changeDistribution(ReturnType min, ReturnType max) {
			if (m_distribution) delete m_distribution;
			m_distribution = new Distribution(min, max);
		}

		ReturnType operator()() {
			return (ReturnType) (*m_distribution)(*m_engine);
		}
	};

	class RealRandomEngine : public AbstractRandomEngine<std::mt19937_64, std::uniform_real_distribution<float>, float> {};
	class BinaryRandomEngine : public AbstractRandomEngine<std::mt19937_64, std::discrete_distribution<float>, float> {};
}