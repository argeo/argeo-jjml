package org.argeo.jjml.llama;

import java.util.function.LongSupplier;

/** A natice llama.cpp sampler. */
public class LlamaCppNativeSampler implements LongSupplier, AutoCloseable, Cloneable {
	private final long pointer;

	private LlamaCppSamplerChain samplerChain = null;

	public LlamaCppNativeSampler(long pointer) {
		this.pointer = pointer;
	}

	private native void doReset();

	private native void doDestroy();

	private native long doClone();

	@Override
	public void close() throws RuntimeException {
		if (samplerChain == null)
			doDestroy();
		else
			throw new IllegalStateException("This sampler cannot be closed as it belong to chain " + samplerChain);
		// TODO remove it from the chain, and then destroy it?
	}

	@Override
	public long getAsLong() {
		return pointer;
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return new LlamaCppNativeSampler(doClone());
	}

	public void reset() {
		doReset();
	}

	void setSamplerChain(LlamaCppSamplerChain currentChain) {
		this.samplerChain = currentChain;
	}

	LlamaCppSamplerChain getSamplerChain() {
		return samplerChain;
	}

}
