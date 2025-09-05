package org.argeo.jjml.llm;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A llama.cpp sampler implemented in Java. Use
 * {@link LlamaCppSamplers#newJavaSampler(LlamaCppJavaSampler)} in order to get
 * the related {@link LlamaCppNativeSampler} to be added to a sampler chain.
 */
public interface LlamaCppJavaSampler {
	/**
	 * Apply sampling.
	 * 
	 * @return the selected index or a negative number if unchanged.
	 */
	long apply(ByteBuffer buf, long size, long selected, boolean sorted);

	/** Does nothing by default. */
	default void accept(int token) {

	}

	/** Does nothing by default. */
	default void reset() {

	}

	default String getName() {
		return getClass().getName();
	}

	/** Trivial Java implementation of a greedy sampler. */
	static class SimpleGreedy implements LlamaCppJavaSampler {

		@Override
		public long apply(ByteBuffer buf, long size, long selected, boolean sorted) {
			ByteBuffer b = buf.duplicate();
			b.order(ByteOrder.nativeOrder());
			long count = 0;
			long res = 0;
			float bestLogit = Float.NEGATIVE_INFINITY;
			while (count < size) {
				b.getInt(); // token
				float logit = b.getFloat();
				b.getFloat(); // probability
				if (count == 0) {
					bestLogit = logit;
				} else if (logit > bestLogit) {
					res = count;
					bestLogit = logit;
				}
				count++;
			}
			return res;
		}

	}
}
